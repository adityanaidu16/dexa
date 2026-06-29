"""Dexa Cartridges — the context compiler.

Turn a static corpus into a small, portable, trained KV cache (a *cartridge*)
that loads into an inference engine as a precomputed prefix. See docs/CARTRIDGES.md.

    from dexa.cartridge import Cartridge, CartridgeCompiler
"""

from dexa.cartridge.artifact import Cartridge

__all__ = ["Cartridge"]

# CartridgeCompiler is torch-heavy; import lazily to keep this package importable
# without torch (e.g. for the artifact format / serving-side tooling).
def __getattr__(name):  # pragma: no cover - thin lazy import
    if name == "CartridgeCompiler":
        from dexa.cartridge.compiler import CartridgeCompiler
        return CartridgeCompiler
    raise AttributeError(name)
