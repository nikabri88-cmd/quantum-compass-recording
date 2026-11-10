#!/usr/bin/env bash
# MAG-S3c pipeline driver: cells and references of the pair, the composite pair, invariant checks, time curves,
# receptor and maps, analysis.
#
#   bash scripts/pipeline.sh smoke        OUT_PARENT [HOURS]   60 directions (composite pair 50), curves to 50 ns,
#                                                              substitute rates; technical facts only, output deleted
#   bash scripts/pipeline.sh blind        OUT_PARENT [HOURS]   every cell on 500 directions, the composite pair on 100,
#                                                              full curves, substitute rates (k_F = 6e6, k_R = 5e5 s^-1,
#                                                              rates of E12 and hops of E13 times 1.2; work order,
#                                                              step 3); technical facts only, output deleted
#   bash scripts/pipeline.sh confirmatory OUT_PARENT [HOURS]   the run (step 5); keeps everything
#
# The confirmatory run starts only when OUT_PARENT/confirmation.json, written by cell 5 of the notebook after the
# answer «запускай», holds that answer, the SHA-256 of every file of the study as it is in this folder, the build
# identity and the marker of the passed blind check of the same build; mag_s3c.py checks all of it again on every
# call with --confirmatory and computes nothing at the rates of the hypotheses without it. The smoke and blind checks exit with 1
# when an invariant fails or the result is malformed. WORKERS sets the number of processes (default: every
# processor). The cells work for at most HOURS per call (default 1) and the call exits with code 3 when it pauses;
# rerun the same command to go on. Finished steps are not redone. Exit code 0: done, 1: failure, 3: paused.
set -u
KIND="$1"; PARENT="$(mkdir -p "$2" && cd "$2" && pwd)"; MAXH="${3:-1.0}"
WORKERS="${WORKERS:-$(nproc)}"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
case "$KIND" in
  smoke)        OUT="$PARENT/smoke_s3c"; GRID=(--grid-half 30); CGRID=(--grid-half 25); CURVE=(--t-max 5e-8)
                RATES=(--blind-rates) ;;
  blind)        OUT="$PARENT/blind_s3c"; GRID=(--grid-half 250); CGRID=(--grid-half 50); CURVE=(); RATES=(--blind-rates) ;;
  confirmatory) OUT="$PARENT/confirmatory_s3c"; GRID=(); CGRID=(); CURVE=()
                RATES=(--confirmatory "$PARENT/confirmation.json")
                python3 -I "$CODE/mag_s3c.py" check-confirmation --confirmatory "$PARENT/confirmation.json" || exit 1
                ;;
  *) echo "first argument: smoke, blind or confirmatory"; exit 2 ;;
esac
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
  if [ $rc -ne 0 ]; then
    if [ "$KIND" = "confirmatory" ]; then tail -5 "$OUT/$name.log"; else grep -E "Error|error|Traceback" "$OUT/$name.log" | tail -3; fi
    exit 1
  fi
}
log "$KIND run in $OUT with $WORKERS workers ($(nproc) processors)"
step pair     "$OUT/run/pair_done.json"   python3 -I mag_s3c.py pair --out "$OUT/run" --workers "$WORKERS" \
     --max-hours "$MAXH" "${GRID[@]}" "${RATES[@]}"
step composite "$OUT/run/composite_done.json" python3 -I mag_s3c.py composite --out "$OUT/run" --workers "$WORKERS" \
     --max-hours "$MAXH" "${CGRID[@]}" "${RATES[@]}"
step checks   "$OUT/run/checks.json"      python3 -I mag_s3c.py checks --out "$OUT/run" --workers "$WORKERS" "${RATES[@]}"
step curves   "$OUT/run/curves.json"      python3 -I mag_s3c.py curves --out "$OUT/run" --workers "$WORKERS" \
     "${CURVE[@]}" "${RATES[@]}"
step receptor "$OUT/run/receptor.json"    python3 -I mag_s3c.py receptor --out "$OUT/run" --workers "$WORKERS"
step analyze  "$OUT/results/results.json" python3 -I analyze_s3c.py --run "$OUT/run" --out "$OUT/results"
log "cells $(find "$OUT/run/cells" -name '*.json' | wc -l), references $(find "$OUT/run/refs" -name '*.json' | wc -l), composite $(find "$OUT/run/composite" -maxdepth 1 -name '*.json' | wc -l)"

if [ "$KIND" != "confirmatory" ]; then
  python3 -I - "$OUT/results/results.json" "$KIND" <<'PY' || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
kind = sys.argv[2]
need = ("valid", "invariants", "expectations", "hypotheses", "comparisons", "multiplicity", "maps", "metadata")
ok_verdicts = {"met", "not_met", "untestable"}
keys = all(k in r for k in need)
hyp = r.get("hypotheses", {})
geoms = {"H1": ["S2", "W_C", "W_D"], "H2": ["W_C", "W_D"], "H3": ["W_C", "W_D"], "H4": ["S2", "W_C", "W_D"],
         "H5": ["S2", "W_C", "W_D"]}
flat = [hyp.get(h, {}).get(g, {}).get("verdict") for h, gs in geoms.items() for g in gs]
inv = {k: v["verdict"] for k, v in r.get("invariants", {}).items()}
exp_ok = all(v["verdict"] in ok_verdicts for v in r.get("expectations", {}).values()) and len(r.get("expectations", {})) == 4
hyp_ok = sorted(hyp) == ["H1", "H2", "H3", "H4", "H5"] and all(v in ok_verdicts for v in flat)
cmp = r.get("comparisons", {})
counts_ok = cmp.get("counted", {}).get("per_hypothesis") == cmp.get("protocol") \
    and cmp.get("counted", {}).get("known_in_advance") == cmp.get("known_protocol")
maps_ok = sorted(r.get("maps", {}), key=lambda k: int(k[1:])) == [f"E{i}" for i in range(1, 15)]
print("result keys present:", keys)
print("invariants:", inv)
print("valid:", r.get("valid"), " grid_ok:", r.get("grid_ok"), " rates_ok:", r.get("rates_ok"), " curves_full:",
      r.get("curves_full"), " cells_complete:", r.get("cells_complete"), " composite_done:", r.get("composite_done"),
      " blas_single_thread:",
      r.get("blas_single_thread"), "(grid, rates and curves differ from the protocol in the checks)")
print("expectations well formed:", exp_ok, " hypotheses well formed:", hyp_ok)
print("comparisons as in the protocol:", counts_ok, " maps E1-E14 present:", maps_ok)
want_grid = {"smoke": (30, [25]), "blind": (250, [50])}[kind]
grid_seen = (r.get("grid_half"), r.get("e13_grid_half"))
blind_rates = {"cry4a": [6.0e6, 5.0e5], "cr2_1e2": [1.2e6, 1.2e8], "cr2_1e3": [1.2e6, 1.2e9]}
rates_seen = r.get("rates_seen", {})
substitute = (rates_seen.get("by_set") == {k: [v] for k, v in blind_rates.items()}
              and rates_seen.get("hop_scale") == [1.2])
print("grids as intended:", grid_seen == want_grid, " substitute rates throughout:", substitute)
ok = (keys and inv and len(inv) == 11 and all(v == "ok" for v in inv.values()) and r.get("cells_complete") is True
      and r.get("pair_done") is True and r.get("composite_done") is True and grid_seen == want_grid and substitute
      and exp_ok and hyp_ok
      and counts_ok and maps_ok and r.get("blas_single_thread") is True
      and (kind != "blind" or r.get("curves_full") is True))
print("check passed" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
PY
fi
log "done"
