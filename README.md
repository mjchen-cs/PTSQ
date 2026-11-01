# PTSQ — supplemental material

Code, traces and experiment logs for *Privacy-Tiered Semantic Query Processing across Device,
Edge, and Cloud LLMs with Accuracy Guarantees* (ICDE 2027 submission).

Every number and figure in the paper is produced by replaying the recorded traces in
`results/traces_main/` with the scripts below; no LLM or API access is needed to reproduce them.

## 1. Setup (about 2 minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -e ".[dev]"
pytest -q                                                 # 25 tests, incl. empirical guarantee checks
```

## 2. Reproduce

| Script | What it does | Time (10 cores) |
|---|---|---|
| `run_small.sh` | unit tests + main comparison (2 seeds) + figures | ~15 min |
| `run_all.sh` | all suites of Section VIII (main, ablation, guarantee with 200 seeds, sensitivity, wrong rules, device model, overheads) + figures and tables | ~8 h |

Outputs go to `results/repro/` and `paper_out/`. The logs we used for the paper are in
`results/exp_v12/` (one row per query, method and seed), so the figures can also be regenerated
without re-running anything:

```bash
python scripts/make_figures.py --exp results/exp_v12 --paper paper_out --traces results/traces_main \
  --live results/live_v2.csv                                 # Figs. 8-11, tables
python scripts/make_design_figs.py --pilot results/pilot/traces_q35 --vague14 results/pilot/traces_vague14 \
  --vague32 results/pilot/traces_e32 --profiles results/profiles.json --main results/exp_v12/main.csv \
  --out paper_out/figs                                       # Figs. 1, 4, 5, 7 (pilot study)
```

## 3. Contents

| Path | Content |
|---|---|
| `ptsq/cascade.py` | certified calibration: configuration grid, union bound (Thm. 1), propose-and-test with the direct recall/precision test (Thm. 2, Eq. 3), learned risk and prefilter |
| `ptsq/privacy.py` | PII detectors, privacy policy, privacy shield (Thm. 4) |
| `ptsq/simulator.py` | discrete-event simulator and monotone escalations (Thm. 3) |
| `ptsq/replay.py` | trace replay of PTSQ and all baselines (Static-Ref, Conf-Cascade, Cert-2Tier, SUPG-JT, BARGAIN-JT) |
| `ptsq/bargain.py` | our re-implementation of BARGAIN's recall-target procedure (betting tests) |
| `ptsq/executor.py`, `ptsq/llm/` | live executor and tier back-ends used on the testbed |
| `scripts/run_experiments.py` | experiment suites; `scripts/make_figures.py` figures and tables |
| `configs/default.yaml` | parameters used in the paper (γ = 0.8, δ = 0.05, B = 40, ...) |
| `results/traces_main/` | one trace per query Q1–Q5 (30,000 tuples each): privacy level, device/edge scores, edge and cloud labels, token counts, measured device latency, keep-masks of the candidate rules (`*.rules.json`) |
| `results/feat/` | review texts used for the TF-IDF features of the learned risk and prefilter |
| `results/profiles.json` | measured tier profiles of the testbed |
| `results/exp_v12/` | raw experiment logs behind every figure and table |
| `results/pilot/` | pilot traces of Section VIII-B (tier agreement, predicate phrasing, rules, calibration example) |
| `results/live_v2.csv` | end-to-end live runs on the testbed (Fig. 10(d)) |
| `data/*.csv.gz` | the full candidate tables (Yelp 150k, Amazon 60k reviews: `rid`, `text`, `privacy`) with injected synthetic personal data; `gunzip` them into `data/` to collect new traces |

## 4. Data

Queries run over public Yelp and Amazon review corpora (`data/`, built by `scripts/prepare_data.py`). The personal data in the texts
(national IDs, bank cards, phone numbers, e-mails) is **synthetic**: checksum-valid values
injected as described in Section VIII-A; no real personal data is included.

## 5. Live runs

Collecting new traces or running end to end requires the three tiers (a Jetson device,
a vLLM edge server and a cloud API key in an environment variable named in
`configs/default.yaml`); see `scripts/collect_trace.py` and `scripts/run_live.py`.
API keys are never read from files.
