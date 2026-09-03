#!/usr/bin/env bash
# Sequential driver: for each image, wait for disk and for any running replay on it, pull, replay all sessions, remove the image.
S=/tmp/claude-0/-home-user-dexa/86b25b67-2276-559a-a54a-f598498ce4d8/scratchpad/traces
cd /home/user/dexa/experiments/speculative-tool-exec
IMAGES=(
 jyangballin/swesmith.x86_64.conan-io_1776_conan.86f29e13
 jyangballin/swesmith.x86_64.pandas-dev_1776_pandas.95280573
 jyangballin/swesmith.x86_64.dask_1776_dask.5f61e423
 jyangballin/swesmith.x86_64.sqlfluff_1776_sqlfluff.50a1c4b6
 jyangballin/swesmith.x86_64.adrienverge_1776_yamllint.8513d9b9
 jyangballin/swesmith.x86_64.benoitc_1776_gunicorn.bacbf8aa
 jyangballin/swesmith.x86_64.pydicom_1776_pydicom.7d361b3d
 jyangballin/swesmith.x86_64.pydata_1776_patsy.a5d16484
 jyangballin/swesmith.x86_64.modin-project_1776_modin.8c7799fd
 jyangballin/swesmith.x86_64.matthewwithanm_1776_python-markdownify.6258f5c3
 jyangballin/swesmith.x86_64.suor_1776_funcy.207a7810
 jyangballin/swesmith.x86_64.knio_1776_dominate.9082227e
 jyangballin/swesmith.x86_64.jaraco_1776_inflect.c079a96a
 jyangballin/swesmith.x86_64.cantools_1776_cantools.0c6a7871
 jyangballin/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
 jyangballin/swesmith.x86_64.pudo_1776_dataset.5c2dc8d3
 jyangballin/swesmith.x86_64.jd_1776_tenacity.0d40e76f
 jyangballin/swesmith.x86_64.luozhouyang_1776_python-string-similarity.115acaac
 jyangballin/swesmith.x86_64.bottlepy_1776_bottle.a8dfef30
 jyangballin/swesmith.x86_64.pycqa_1776_flake8.cf1542ce
)
for img in "${IMAGES[@]}"; do
  short=$(echo "$img" | sed -E 's/.*1776_([^.]+)\..*/\1/')
  # wait for the previous batch (flake8 or otherwise) to release CPU/disk
  while pgrep -f "replay.py --sessions" >/dev/null; do sleep 20; done
  while [ "$(df --output=avail -BG / | tail -1 | tr -dc '0-9')" -lt 9 ]; do
    for old in $(docker images --format '{{.Repository}}' | grep swesmith | grep -v "$img"); do docker rmi -f "$old" >/dev/null 2>&1; done
    sleep 5
    [ "$(df --output=avail -BG / | tail -1 | tr -dc '0-9')" -lt 9 ] && { echo "$(date -u +%T) low disk, waiting"; sleep 30; }
  done
  if ! docker image inspect "$img:latest" >/dev/null 2>&1; then
    ok=0
    for attempt in 1 2 3 4 5 6 7 8; do
      echo "$(date -u +%T) PULL $img (attempt $attempt)"
      if docker pull "$img:latest" >/dev/null 2>&1; then ok=1; break; fi
      sleep $((60 * attempt))
    done
    [ "$ok" = 1 ] || { echo "$(date -u +%T) PULL FAILED $img"; continue; }
  fi
  echo "$(date -u +%T) REPLAY $short"
  python3 replay.py --sessions $S/thoughtworks_adapted.json --instances $S/swesmith_instances.json --image "$img" --out "runs/$short.jsonl" --worker wb > "runs/$short.log" 2>&1
  echo "$(date -u +%T) DONE $short $(wc -l < runs/$short.jsonl) sessions"
  docker rmi -f "$img:latest" >/dev/null 2>&1
done
echo "$(date -u +%T) ALL DONE"
