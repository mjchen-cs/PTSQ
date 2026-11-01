#!/usr/bin/env bash
# End-to-end DRY RUN on synthetic traces (no GPU, no API keys). Outputs go to results_mock/ and
# paper_mock/ so they can never be confused with real results.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/make_mock_traces.py --out results_mock/traces --n 20000
if python -c "import torch" 2>/dev/null; then
  python scripts/train_router.py --traces results_mock/traces --config configs/mock.yaml --episodes 400 --out results_mock/router.pt
fi
python scripts/run_experiments.py --suite all --traces results_mock/traces --config configs/mock.yaml \
  --router results_mock/router.pt --seeds 3 --guarantee-seeds 40 --jobs 8 --out results_mock/exp
mkdir -p paper_mock
python scripts/make_figures.py --exp results_mock/exp --paper paper_mock
echo "dry run finished: see results_mock/ and paper_mock/ (SYNTHETIC - do not report)"
