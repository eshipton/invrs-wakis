#!/usr/bin/env bash
# sweep_gpu.sh — GPU variant of autoresearch-cloud bin/sweep.sh (Cloud Batch).
#
#   scripts/sweep_gpu.sh <nshards> <machine> [--minutes MIN] [--tag T] [--standard] -- <script args>
#
# Builds the job with `DRY_RUN=1 sweep.sh submit` (same task script, repo commit and
# upload of /work/out), then patches it: the task installs the 'gpu' extra
# (uv sync --extra gpu), and the VM gets one NVIDIA L4 (installGpuDrivers) on a
# g2-standard-* machine. The script run is scripts/run_challenge.py.
set -euo pipefail
KIT=~/QBoltz-Code/autoresearch-cloud
REPO="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:?nshards}"; MACHINE="${2:?machine (g2-standard-8/16/32)}"; shift 2
MIN=240; TAG=gpu; SPOT=1; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --minutes) MIN="$2"; shift 2;; --tag) TAG="$2"; shift 2;; --standard) SPOT=0; shift;;
    --) shift; PASS=("$@"); break;; *) echo "unknown option $1" >&2; exit 1;;
  esac
done
out=$(DRY_RUN=1 "$KIT/bin/sweep.sh" submit "$REPO" scripts/run_challenge.py "$N" --machine "$MACHINE" \
      --minutes "$MIN" --tag "$TAG" $([ $SPOT = 0 ] && echo --standard) -- "${PASS[@]}" 2>&1)
echo "$out" | tail -2
JSON=$(echo "$out" | sed -n 's/.*job.json at \(.*\), nothing submitted.*/\1/p')
[ -f "$JSON" ] || { echo "no job.json produced" >&2; exit 1; }
read -r RUN_ID BUCKET < <(python3 -c "import json,sys; e=json.load(open(sys.argv[1]))['taskGroups'][0]['taskSpec']['environment']['variables']; print(e['RUN_ID'], e['BUCKET'])" "$JSON")
TMP=$(mktemp -d)
gcloud storage cp "gs://$BUCKET/sweeps/$RUN_ID/task.sh" "$TMP/task.sh" --quiet
sed -i.bak 's/uv sync --quiet/uv sync --quiet --extra gpu/' "$TMP/task.sh"
grep -q -- "--extra gpu" "$TMP/task.sh" || { echo "task.sh patch failed" >&2; exit 1; }
gcloud storage cp "$TMP/task.sh" "gs://$BUCKET/sweeps/$RUN_ID/task.sh" --quiet
python3 - "$JSON" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
inst = j["allocationPolicy"]["instances"][0]
inst["installGpuDrivers"] = True
inst["policy"]["accelerators"] = [{"type": "nvidia-l4", "count": 1}]
inst["policy"]["bootDisk"]["sizeGb"] = 100
j["labels"]["gpu"] = "l4"
json.dump(j, open(sys.argv[1], "w"), indent=1)
PY
. "$KIT/config.env"
JOB="$(printf 'sw-%s' "$RUN_ID" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-\n' '-' | cut -c1-60 | sed 's/-*$//')"
gcloud batch jobs submit "$JOB" --location "$REGION" --config "$JSON" --quiet >/dev/null
echo "submitted $JOB ($N tasks on $MACHINE + L4)"
echo "$RUN_ID"
