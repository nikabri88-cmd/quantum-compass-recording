#!/usr/bin/env bash
# MAG-S3b pipeline driver: pair grid, receptor maps, E3 averages, records, blind estimates, E4, analysis.
#
#   bash scripts/pipeline.sh smoke        OUT_PARENT CAL_DIR           small grid, birds 9411-9412, output deleted
#   bash scripts/pipeline.sh blind        OUT_PARENT CAL_DIR [HOURS]   full grid at substitute rates (BLIND_RATES of
#                                                                      mag_s3b.py), birds 9401-9402 (work order, step
#                                                                      3); technical facts only, output deleted
#   bash scripts/pipeline.sh confirmatory OUT_PARENT CAL_DIR [HOURS]   the run (step 5); keeps everything
#
# CAL_DIR holds C1.json and C3.json of the calibration on the final code (H6 and H5 are gated on them). WORKERS
# sets the number of processes (default: every processor). The pair grid works for at most HOURS per call (default
# 1) and the call exits with code 3 when it pauses; rerun the same command to go on. Finished steps are not redone.
# The records folder never holds the truth; estimate.py refuses a folder that does, and analyze_s3b.py opens the
# truth only after I7. Exit code 0: done, 1: failure, 3: paused.
set -u
KIND="$1"; PARENT="$(mkdir -p "$2" && cd "$2" && pwd)"; CAL="$3"; MAXH="${4:-1.0}"
WORKERS="${WORKERS:-$(nproc)}"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
case "$KIND" in
  smoke)        OUT="$PARENT/smoke_9411_9412"; GRID=(--grid-half 30 --blind-rates); SEEDS=(--seeds 9411-9412)
                E3SEEDS=(--seeds 7001-7003) ;;
  blind)        OUT="$PARENT/blind_9401_9402"; GRID=(--blind-rates); SEEDS=(--seeds 9401-9402); E3SEEDS=() ;;
  confirmatory) OUT="$PARENT/confirmatory_s3b"; GRID=(); SEEDS=(); E3SEEDS=() ;;
  *) echo "first argument: smoke, blind or confirmatory"; exit 2 ;;
esac
CAL="$(cd "$CAL" && pwd)"
for f in C1.json C3.json; do [ -f "$CAL/$f" ] || { echo "missing $CAL/$f"; exit 1; }; done
mkdir -p "$OUT"
cd "$CODE"
if [ "$KIND" != "confirmatory" ]; then
  # the check output goes on every exit except a pause, which keeps it for the rerun
  trap 'rc=$?; if [ $rc -ne 3 ]; then rm -rf "$OUT"; echo "check output deleted (exit $rc)"; fi' EXIT
fi
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
step() {   # $1 = name, $2 = file whose presence marks the step as done, rest = command
  local name="$1" done="$2"; shift 2
  [ -e "$done" ] && return 0
  "$@" >> "$OUT/$name.log" 2>&1
  local rc=$?
  log "$name exit $rc"
  if [ $rc -eq 3 ]; then log "$name paused at the time limit; rerun the same command"; exit 3; fi
  if [ $rc -ne 0 ]; then tail -5 "$OUT/$name.log"; exit 1; fi
}
log "$KIND run in $OUT with $WORKERS workers ($(nproc) processors)"
step physics  "$OUT/pair/invariants_physics.json" python3 -I mag_s3b.py physics --out "$OUT/pair" \
     --workers "$WORKERS" --max-hours "$MAXH" "${GRID[@]}"
step receptor "$OUT/pair/receptor.json"           python3 -I mag_s3b.py receptor --out "$OUT/pair"
step e3geo    "$OUT/records/e3_geometry.json"     python3 -I mag_s3b.py e3geo --records "$OUT/records" "${E3SEEDS[@]}"
step generate "$OUT/records/manifest.json"        python3 -I mag_s3b.py generate --records "$OUT/records" \
     --truth "$OUT/truth" --workers "$WORKERS" "${SEEDS[@]}"
step estimate "$OUT/estimates/estimates.sha256"   python3 -I estimate.py --records "$OUT/records" \
     --out "$OUT/estimates" --workers "$WORKERS"
step e4       "$OUT/pair/e4.json"                 python3 -I mag_s3b.py e4 --out "$OUT/pair" --workers "$WORKERS" \
     "${SEEDS[@]}"
step analyze  "$OUT/results/results.json"         python3 -I analyze_s3b.py --pair "$OUT/pair" --calibration "$CAL" \
     --records "$OUT/records" --estimates "$OUT/estimates" --truth "$OUT/truth" --out "$OUT/results"
log "records $(find "$OUT/records" -name '*.npz' | wc -l), cells $(find "$OUT/pair/cells" -name '*.json' | wc -l)"

if [ "$KIND" != "confirmatory" ]; then
  python3 -I - "$OUT/results/results.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
need = ("valid", "invariants", "expectations", "hypotheses", "multiplicity", "maps", "metadata")
ok = {"met", "not_met", "untestable"}
print("result keys present:", all(k in r for k in need))
print("invariants:", {k: v["verdict"] for k, v in r["invariants"].items()})
print("valid:", r["valid"], " grid_ok:", r["grid_ok"], " rates_ok:", r["rates_ok"], "(False at the substitute rates)")
print("expectations well formed:", sorted(r["expectations"]), all(v.get("verdict") in ok for v in r["expectations"].values()))
print("hypotheses well formed:", sorted(r["hypotheses"]), all(v.get("verdict") in ok for v in r["hypotheses"].values()))
print("maps:", sorted(r["maps"]))
EOF
fi
log "done"
