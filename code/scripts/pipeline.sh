#!/usr/bin/env bash
# MAG-S3 pipeline driver: physics, stage 1, analyze.py points, stage 2, analyze.py final.
#
#   bash scripts/pipeline.sh blind        OUT_PARENT [MAX_HOURS]
#   bash scripts/pipeline.sh confirmatory OUT_PARENT [MAX_HOURS]
#
# blind         birds 9301-9302 in both stages (work order, step 3). Prints only technical facts (exit codes, file
#               counts, input checks, invariant verdicts, presence of result keys) and deletes the output without
#               showing any expectation or hypothesis verdict.
# confirmatory  birds 1-10 (stage 1) and 101-200 (stage 2); keeps everything.
#
# WORKERS sets the number of processes (default: every processor). Each call works for at most MAX_HOURS per
# simulation stage (default 1.8) and exits with code 3 when it pauses at the limit; rerun the same command to go on.
# Finished seed files are never recomputed. Exit code 0: done, 1: failure, 3: paused.
set -u
KIND="$1"; PARENT="$2"; MAXH="${3:-1.8}"
WORKERS="${WORKERS:-$(nproc)}"
MEM_GB=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
CAP=$(( MEM_GB * 10 / 12 ))                       # about 1.2 GB per worker; the largest job needs 0.7 GB
if [ "$CAP" -ge 1 ] && [ "$WORKERS" -gt "$CAP" ]; then WORKERS=$CAP; fi
CODE="$(cd "$(dirname "$0")/.." && pwd)"
case "$KIND" in
  blind)        OUT="$PARENT/blind_9301_9302"; SIM_SEEDS1=(--seeds 9301-9302); SIM_SEEDS2=(--seeds 9301-9302)
                ANA_SEEDS=(--seeds1 9301-9302 --seeds2 9301-9302) ;;
  confirmatory) OUT="$PARENT/confirmatory_v04"; SIM_SEEDS1=(); SIM_SEEDS2=(); ANA_SEEDS=() ;;
  *) echo "first argument: blind or confirmatory"; exit 2 ;;
esac
mkdir -p "$OUT/s1" "$OUT/s2"
cd "$CODE"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
log "$KIND run in $OUT with $WORKERS workers ($(nproc) processors, ${MEM_GB} GB memory available)"

if [ ! -f "$OUT/s1/physics_s3.json" ]; then
  python3 -I mag_s3.py physics --out "$OUT/s1" > "$OUT/physics.log" 2>&1
  rc=$?; log "physics exit $rc"
  [ $rc -ne 0 ] && { tail -5 "$OUT/physics.log"; exit 1; }
fi

run_stage() {   # $1 = stage1|stage2, rest = arguments
  local st="$1"; shift
  [ -f "$OUT/$st.done" ] && return 0
  python3 -I mag_s3.py "$st" "$@" --workers "$WORKERS" --max-hours "$MAXH" >> "$OUT/$st.log" 2>&1
  local rc=$?
  log "$st exit $rc"
  if [ $rc -ne 0 ]; then log "$st failed"; tail -5 "$OUT/$st.log"; exit 1; fi
  local dir="$OUT/s${st#stage}" meta
  meta=$(ls -t "$dir"/meta_"$st"_*.json 2>/dev/null | head -1)
  if [ -z "$meta" ] || ! grep -q '"complete": true' "$meta"; then
    log "$st paused at the time limit: $(grep -E '^[0-9]+/[0-9]+ jobs' "$OUT/$st.log" | tail -1)"; exit 3
  fi
  touch "$OUT/$st.done"
}

run_stage stage1 --out "$OUT/s1" "${SIM_SEEDS1[@]}"
log "stage1 files $(find "$OUT/s1" -name 'seed_*.json' | wc -l) in $(find "$OUT/s1" -mindepth 1 -maxdepth 1 -type d | wc -l) cells"

if [ ! -f "$OUT/s1/working_points.json" ]; then
  python3 -I analyze.py points "$OUT/s1" "${ANA_SEEDS[@]:0:2}" > "$OUT/points.log" 2>&1
  rc=$?; log "analyze points exit $rc"
  if [ $rc -ne 0 ]; then
    python3 -I - "$OUT/points.log" <<'EOF'
import json, sys
text = open(sys.argv[1]).read()
try:
    print("refused:", json.loads(text).get("refused"))
except Exception:
    print(text[-2000:])
EOF
    exit 1
  fi
fi

run_stage stage2 --out "$OUT/s2" "${SIM_SEEDS2[@]}" --points "$OUT/s1/working_points.json"
log "stage2 files $(find "$OUT/s2" -name 'seed_*.json' | wc -l) in $(find "$OUT/s2" -mindepth 1 -maxdepth 1 -type d | wc -l) cells"

python3 -I analyze.py final "$OUT/s1" "$OUT/s2" "${ANA_SEEDS[@]}" > "$OUT/final.log" 2>&1
rc=$?; log "analyze final exit $rc"
[ $rc -ne 0 ] && { tail -3 "$OUT/final.log"; exit 1; }

if [ "$KIND" = "blind" ]; then
  python3 -I - "$OUT/s2/results.json" "$OUT/s1/working_points.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
need = ("valid", "input_issues", "confirmatory", "seeds", "invariants", "expectations", "working_points", "hypotheses",
        "requirements", "exploratory", "multiplicity", "analysis_sha256", "code")
print("result keys present:", all(k in r for k in need))
print("input checks:", r["input_issues"] or "no problems")
print("invariants:", {k: v["verdict"] for k, v in r["invariants"].items()}, "I1 by", r["invariants"]["I1"].get("how"))
print("valid:", r["valid"], " confirmatory:", r["confirmatory"])
ok = {"met", "not met", "untestable"}
print("expectations well formed:", sorted(r["expectations"]), all(v.get("verdict") in ok for v in r["expectations"].values()))
print("hypotheses well formed:", sorted(r["hypotheses"]), all(v.get("verdict") in ok for v in r["hypotheses"].values()))
print("exploratory maps:", sorted(r["exploratory"]))
print("working point keys:", sorted(json.load(open(sys.argv[2]))))
EOF
  log "deleting the blind output"
  rm -rf "$OUT"
fi
log "done"
