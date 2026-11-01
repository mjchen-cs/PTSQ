#!/usr/bin/env bash
# Quick check (~15 min on 10 cores): tests, the main comparison with 2 seeds, figures.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=${JOBS:-$(nproc)}
pytest -q
python scripts/run_experiments.py --suite main --seeds 2 --traces results/traces_main \
  --features results/feat --profiles results/profiles.json --jobs "$JOBS" --out results/repro_small
python scripts/make_figures.py --exp results/repro_small --paper paper_out_small --traces results/traces_main
echo "done: results/repro_small/main.csv, paper_out_small/figs/fig_main.pdf"
