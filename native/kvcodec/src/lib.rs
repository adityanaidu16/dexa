//! Native accelerator for Dexa's memory-mapped KV blob format.
//!
//! What this buys over the pure-numpy path (`dexa/session/blob.py`):
//! the *save* side has to pack every fp32 KV element to the model's storage
//! precision (bf16/fp16) and stream the result to disk. numpy does that
//! single-threaded, materializing a full intermediate copy via `.tobytes()`.
//! This crate packs the layers **in parallel** (rayon) straight into the output
//! byte buffer and writes header + payload in one pass. The load side stays in
//! Python because it is already zero-copy (`mmap`) — there is nothing for native
//! code to beat there.
//!
//! Correctness contract: the bytes written here are **identical** to the numpy
//! fallback. bf16 uses round-to-nearest-even on the fp32 bit pattern (matching
//! `dexa.session.state._f32_to_bf16_bits`); fp16 uses the standard IEEE cast;
//! fp32 is a raw copy. The load path (`load_kvcache_blob`) reads either writer's
//! output — this only changes *how fast* the file is produced.
//!
//! Build (needs a Rust toolchain; not compiled in the laptop/CI env):
//!     pip install maturin
//!     maturin develop -m native/kvcodec/Cargo.toml --release
//! after which `from dexa.session import kvcodec` succeeds and
//! `save_kvcache_blob` uses it automatically.

use std::fs::File;
use std::io::{BufWriter, Write};

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

/// fp32 -> bf16 bits (u16), round-to-nearest-even. Matches the numpy helper.
#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    // round-to-nearest-even on the discarded low 16 bits.
    let bias = ((u >> 16) & 1) + 0x7fff;
    (((u as u64 + bias as u64) >> 16) as u16).to_le()
}

/// Bytes-per-element for a storage tag.
fn elem_size(store_dtype: &str) -> PyResult<usize> {
    match store_dtype {
        "float32" => Ok(4),
        "float16" | "bfloat16" => Ok(2),
        other => Err(PyValueError::new_err(format!("unknown store_dtype {other:?}"))),
    }
}

/// Pack one fp32 slab into `out` (already sized) at the given storage precision.
fn pack_into(src: &[f32], store_dtype: &str, out: &mut [u8]) {
    match store_dtype {
        "float32" => {
            // little-endian raw copy, chunked for parallelism.
            out.par_chunks_mut(4)
                .zip(src.par_iter())
                .for_each(|(dst, &v)| dst.copy_from_slice(&v.to_le_bytes()));
        }
        "float16" => {
            out.par_chunks_mut(2)
                .zip(src.par_iter())
                .for_each(|(dst, &v)| dst.copy_from_slice(&half_from_f32(v).to_le_bytes()));
        }
        "bfloat16" => {
            out.par_chunks_mut(2)
                .zip(src.par_iter())
                .for_each(|(dst, &v)| dst.copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes()));
        }
        _ => unreachable!("validated by elem_size"),
    }
}

/// IEEE-754 half from f32 (round-to-nearest-even). Small, dependency-free.
#[inline]
fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        // subnormal/underflow -> flush toward zero (KV values rarely hit this).
        return sign;
    } else if exp >= 0x1f {
        return sign | 0x7c00; // inf/overflow
    }
    // round-to-nearest-even on the 13 discarded mantissa bits.
    let mant10 = (mant >> 13) as u16;
    let round = (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (mant10 & 1) != 0);
    let mut out = sign | ((exp as u16) << 10) | mant10;
    if round {
        out += 1;
        if (out & 0x7c00) == 0x7c00 {
            exp += 1; // carry into exponent handled by the add; guard inf
        }
    }
    out
}

/// Pack the per-layer fp32 keys/values and write the full blob
/// (magic + hdr_len + header + pad + payload) to `path`.
///
/// `keys`/`values` are lists of C-contiguous fp32 arrays (one per layer),
/// identical shape. Byte-for-byte compatible with the numpy fallback.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn save_blob(
    py: Python<'_>,
    path: &str,
    magic: &Bound<'_, PyBytes>,
    header: &Bound<'_, PyBytes>,
    store_dtype: &str,
    keys: Vec<PyReadonlyArrayDyn<'_, f32>>,
    values: Vec<PyReadonlyArrayDyn<'_, f32>>,
    align: usize,
) -> PyResult<()> {
    let esz = elem_size(store_dtype)?;
    let magic = magic.as_bytes().to_vec();
    let header = header.as_bytes().to_vec();

    // Gather slices (fp32) for every layer's K then V, in on-disk order.
    let mut slabs: Vec<&[f32]> = Vec::with_capacity(keys.len() + values.len());
    for k in &keys {
        slabs.push(k.as_slice()?);
    }
    for v in &values {
        slabs.push(v.as_slice()?);
    }
    let total_elems: usize = slabs.iter().map(|s| s.len()).sum();

    // Pack all slabs into one contiguous payload buffer, in parallel, without the GIL.
    let payload = py.allow_threads(|| {
        let mut payload = vec![0u8; total_elems * esz];
        // Split the output buffer per-slab, then pack each slab in parallel.
        let mut offset = 0usize;
        let mut regions: Vec<(&[f32], &mut [u8])> = Vec::with_capacity(slabs.len());
        let mut rest = payload.as_mut_slice();
        for s in &slabs {
            let n = s.len() * esz;
            let (head, tail) = rest.split_at_mut(n);
            regions.push((s, head));
            rest = tail;
            offset += n;
        }
        debug_assert_eq!(offset, payload.len());
        regions
            .into_par_iter()
            .for_each(|(src, dst)| pack_into(src, store_dtype, dst));
        payload
    });

    let f = File::create(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut w = BufWriter::new(f);
    let mut written = 0usize;
    let mut put = |w: &mut BufWriter<File>, b: &[u8], written: &mut usize| -> PyResult<()> {
        w.write_all(b).map_err(|e| PyValueError::new_err(e.to_string()))?;
        *written += b.len();
        Ok(())
    };
    put(&mut w, &magic, &mut written)?;
    put(&mut w, &(header.len() as u64).to_le_bytes(), &mut written)?;
    put(&mut w, &header, &mut written)?;
    let pad = (align - (written % align)) % align;
    if pad > 0 {
        put(&mut w, &vec![0u8; pad], &mut written)?;
    }
    put(&mut w, &payload, &mut written)?;
    w.flush().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

#[pymodule]
fn kvcodec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_blob, m)?)?;
    m.add("__doc__", "Native (Rust) accelerator for Dexa KV blob save/pack.")?;
    Ok(())
}
