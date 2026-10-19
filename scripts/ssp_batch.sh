#!/usr/bin/env bash
# SSP pilot on Cloud Batch: one spec per n2-standard-2 machine; code shipped as
# a tarball of the committed `ssp` branch (sweep.sh needs a pushed commit).
#   scripts/ssp_batch.sh <tag> <spec> [<spec> ...]   (spec = challenge:seed:arm)
set -euo pipefail
. ~/QBoltz-Code/autoresearch-cloud/config.env
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
TAG="$1"; shift; SPECS=("$@"); N=${#SPECS[@]}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%d-%H%M)-$TAG"
TMP=$(mktemp -d)
git -C "$REPO" archive --format=tar.gz -o "$TMP/code.tgz" HEAD
COMMIT=$(git -C "$REPO" rev-parse HEAD)
printf '%s\n' "${SPECS[@]}" > "$TMP/specs.txt"
cat > "$TMP/task.sh" <<'TASK'
#!/usr/bin/env bash
set -uo pipefail
IDX="${BATCH_TASK_INDEX:-0}"
OUT="gs://${BUCKET}/sweeps/${RUN_ID}/shard_${IDX}"
mkdir -p /work/out /work/repo
log() { printf '[%s] shard %s: %s\n' "$(date -u +%H:%M:%S)" "$IDX" "$*" | tee -a /work/out/task.log; }
gcloud storage cp "gs://${BUCKET}/sweeps/${RUN_ID}/code.tgz" "gs://${BUCKET}/sweeps/${RUN_ID}/specs.txt" /work/ --quiet
tar -xzf /work/code.tgz -C /work/repo && cd /work/repo
SPEC="$(sed -n "$((IDX + 1))p" /work/specs.txt)"
log "spec $SPEC commit $COMMIT; uv sync"
uv sync --quiet || log "uv sync failed"
start=$(date +%s)
timeout "${MINUTES}m" uv run python scripts/run_challenge.py --config configs/round2.json \
  --specs "$SPEC" --outdir /work/out --machine n2-standard-2 --jobs-per-task 1 --no-render \
  >> /work/out/task.log 2>&1
rc=$?
log "exit ${rc} after $(( $(date +%s) - start )) s on $(nproc) cpus"
gcloud storage cp -r /work/out/* "$OUT/" --quiet
exit $rc
TASK
MINUTES=600
python3 - "$TMP/job.json" <<PY
import json, sys
job = {"taskGroups": [{"taskCount": $N, "parallelism": $N, "taskSpec": {
  "runnables": [{"container": {"imageUri": "$IMAGE", "entrypoint": "/bin/bash",
     "commands": ["-c", "gcloud storage cp gs://$BUCKET/sweeps/$RUN_ID/task.sh /work/task.sh --quiet && bash /work/task.sh"]}}],
  "maxRunDuration": "$(( (MINUTES + 30) * 60 ))s", "maxRetryCount": 2,
  "lifecyclePolicies": [{"action": "RETRY_TASK", "actionCondition": {"exitCodes": [50001]}}],
  "environment": {"variables": {"RUN_ID": "$RUN_ID", "BUCKET": "$BUCKET", "COMMIT": "$COMMIT", "MINUTES": "$MINUTES"}}}}],
 "allocationPolicy": {"instances": [{"policy": {"machineType": "n2-standard-2", "provisioningModel": "SPOT",
     "bootDisk": {"sizeGb": 40}}}], "serviceAccount": {"email": "$SA_EMAIL"}},
 "logsPolicy": {"destination": "CLOUD_LOGGING"},
 "labels": {"autoresearch": "sweep", "tag": "ssp-pilot"}}
json.dump(job, open(sys.argv[1], "w"), indent=1)
PY
for f in code.tgz specs.txt task.sh job.json; do gcloud storage cp "$TMP/$f" "gs://$BUCKET/sweeps/$RUN_ID/$f" --quiet; done
JOB="$(printf 'sw-%s' "$RUN_ID" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-\n' '-' | cut -c1-60 | sed 's/-*$//')"
gcloud batch jobs submit "$JOB" --location "$REGION" --config "$TMP/job.json" --quiet >/dev/null
echo "submitted $JOB ($N tasks, commit ${COMMIT:0:8})"; echo "$RUN_ID"
