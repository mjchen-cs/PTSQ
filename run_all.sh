#!/usr/bin/env bash
# All suites of Section VIII (~8 h on 10 cores), then every figure and table.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=${JOBS:-$(nproc)}
pytest -q
python scripts/run_experiments.py --suite main ablation guarantee sensitivity bad_rules device_model overhead \
  --traces results/traces_main --features results/feat --profiles results/profiles.json \
  --jobs "$JOBS" --out results/repro
python scripts/make_figures.py --exp results/repro --paper paper_out --traces results/traces_main --live results/live_v2.csv
python scripts/make_design_figs.py --pilot results/pilot/traces_q35 --vague14 results/pilot/traces_vague14 \
  --vague32 results/pilot/traces_e32 --profiles results/profiles.json --main results/repro/main.csv --out paper_out/figs
echo "done: results/repro/*.csv, paper_out/figs, paper_out/tables"
