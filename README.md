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
| `scripts/collect_trace.py`, `run_live.py`, `profile_tiers.py`, `export_texts.py`, `merge_cache.py`, `batch_label.py` | testbed scripts (Section 5) |
| `configs/default.yaml` | parameters used in the paper (γ = 0.8, δ = 0.05, B = 40, ...) and the three tiers |
| `configs/live.yaml` | the same for live runs on the Jetson (smaller device batches, 15 s cloud timeout) |
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

## 5. Running on the device–edge–cloud testbed

Sections 1–4 need no hardware. This section describes how the traces in `results/traces_main/`
and the live runs in `results/live_v2.csv` were produced, and how to collect new ones.

### 5.1 Roles of the three machines

| Tier | Hardware in the paper | Software | What runs there |
|---|---|---|---|
| Device | NVIDIA Jetson Orin Nano Super 8 GB (JetPack 6.2, CUDA 12.6, Python 3.10) | PyTorch 2.5 (NVIDIA wheel) + Transformers | device model Qwen2.5-1.5B-Instruct (FP16); PII detection; the PTSQ orchestrator of live runs (`run_live.py`) |
| Edge | Workstation with an RTX 5090 32 GB (Windows 11 + WSL2 Ubuntu 22.04, CUDA 12.9) | vLLM, OpenAI-compatible server on port 8000 | edge model Qwen2.5-32B-Instruct-AWQ; also trace replay, experiments and figures |
| Cloud | SiliconFlow API (any OpenAI-compatible provider works) | — | reference model Qwen3.5-122B-A10B (thinking disabled), per-account rate limit |

Data flow: in live runs the Jetson is the client of both remote tiers. It sends MEDIUM and LOW
tuples to the edge server over the LAN and only LOW tuples to the cloud API. Privacy levels and
permitted tiers are set in `privacy:` of `configs/default.yaml`; the privacy shield blocks any
request that would violate them. All three tiers answer the same Yes/No prompt
(`ptsq/llm/prompts.py`); every response is cached in `results/cache.sqlite`.

Throughout, `<EDGE_IP>` is the LAN address of the edge server and `<JETSON_IP>` that of the
Jetson (192.168.0.36 and 192.168.0.34 in our setup). Prompts marked **[Edge]** run in the WSL2
shell of the workstation, **[Device]** in an SSH session on the Jetson.

### 5.2 One-time setup

**Edge server [Edge]**

1. On Windows, enable mirrored networking so that the Jetson reaches WSL2 directly, and open
   port 8000 (PowerShell as administrator):
   ```powershell
   # %USERPROFILE%\.wslconfig must contain:  [wsl2]  networkingMode=mirrored
   New-NetFirewallRule -DisplayName "vLLM-8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
   wsl --shutdown
   ```
   On a native Linux server this step is not needed.
2. Two Python environments, one for vLLM and one for PTSQ:
   ```bash
   sudo apt install -y python3-venv python3-pip build-essential iperf3 tmux
   python3 -m venv ~/venvs/vllm && ~/venvs/vllm/bin/pip install -U pip vllm
   python3 -m venv ~/venvs/ptsq && source ~/venvs/ptsq/bin/activate
   pip install -U pip && pip install -e ".[rl,data,dev]" && pytest -q
   ```
   vLLM compiles some kernels at start-up, so the CUDA toolkit (`nvcc`) must be on `PATH`.
3. If an HTTP proxy is configured, exclude the LAN addresses explicitly (wildcards such as
   `192.168.*` are not honoured):
   `export no_proxy="$no_proxy,<EDGE_IP>,<JETSON_IP>" NO_PROXY="$no_proxy"`.

**Device [Device]**

1. Maximum power mode and clocks (the mode number of MAXN SUPER is shown by `nvpmodel -q`):
   ```bash
   sudo nvpmodel -m <MAXN_SUPER> && sudo jetson_clocks
   ```
2. Install NVIDIA's PyTorch wheel for JetPack 6 *before* PTSQ, then PTSQ with the
   Jetson-compatible pins:
   ```bash
   sudo apt install -y python3-venv libopenblas-dev iperf3 tmux
   python3 -m venv ~/venvs/ptsq && source ~/venvs/ptsq/bin/activate && pip install -U pip "numpy<2"
   pip install --no-cache <torch-2.5 wheel for JetPack 6 from developer.download.nvidia.com>
   pip install -e ".[device,jetson,data]"
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
   ```
   If it prints `False`, a CPU wheel from PyPI has replaced NVIDIA's; reinstall the NVIDIA wheel.
3. The device model is downloaded from Hugging Face on first use
   (`tiers.device.model_path`; set `HF_ENDPOINT` to a mirror if needed).

**Cloud API key (on every machine that calls the cloud)**

The key is read only from the environment variable named in `tiers.cloud.api_key_env`; it is
never written to any file of this repository:
```bash
echo 'export SILICONFLOW_API_KEY=<your key>' >> ~/.bashrc && source ~/.bashrc
```

**Configuration**

Copy the code to the Jetson (`scp -r . nvidia@<JETSON_IP>:~/ptsq`) and keep `configs/` identical
on both machines. Check in `configs/default.yaml`:

- `tiers.edge.base_url: http://<EDGE_IP>:8000/v1` and `tiers.edge.model` equal to the model vLLM serves;
- `tiers.cloud.model`, `price_in`, `price_out` (CNY per million tokens) against the provider's current price list;
- `tiers.cloud.tpm_limit` / `rpm_limit` slightly below your account's limits; the cloud is the
  bottleneck tier, and the paper's main setting uses the entry-level limit (20,000 TPM, 1,000 RPM).

`configs/live.yaml` is the same configuration with smaller device batches (CPU and GPU share 8 GB
during live runs) and a 15 s cloud timeout; use it for the live runs below.

### 5.3 Start the edge model [Edge]

Keep this running in its own terminal (tmux) during trace collection and live runs:
```bash
source ~/venvs/vllm/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --host 0.0.0.0 --port 8000 \
  --quantization awq_marlin --max-model-len 4096 --gpu-memory-utilization 0.90 --generation-config vllm
```
Wait for `Application startup complete`, then check from the Jetson:
```bash
curl http://<EDGE_IP>:8000/v1/models                      # [Device]
```

### 5.4 Measure the network and the tier profiles

The simulator used for replay needs the latency of each tier and the network between them.

1. Network [Edge: `iperf3 -s`], then [Device]:
   ```bash
   iperf3 -c <EDGE_IP> -t 10                                # LAN uplink, Mbps
   ping -c 20 <EDGE_IP>                                     # LAN RTT
   curl -o /dev/null -s -w "%{time_connect}\n" https://api.siliconflow.cn   # cloud RTT
   ```
   and measure the Internet uplink of the Jetson (e.g. `speedtest-cli --secure`).
2. Tier profiles [Device]; the script fits latency = fixed + per-token × tokens and writes
   `results/profiles.json` (an existing file is updated, not overwritten):
   ```bash
   python scripts/profile_tiers.py --tiers device edge cloud \
     --edge-bw <LAN Mbps> --cloud-bw <uplink Mbps> --rtt-edge <LAN RTT s> --rtt-cloud <cloud RTT s>
   ```
   In our setup: `--edge-bw 90 --cloud-bw 23.7 --rtt-edge 0.002 --rtt-cloud 0.08`.

### 5.5 Prepare the data

```bash
gunzip -k data/yelp150k.csv.gz data/amazon60k.csv.gz      # tables used by configs/queries_example.json
```
To build tables from other public corpora (synthetic, checksum-valid PII is injected; 10% HIGH
and 30% MEDIUM by default):
```bash
python scripts/prepare_data.py --hf Yelp/yelp_review_full --n 30000 --out data/my_table.csv
```
Queries are defined in `configs/queries_example.json` (`qid`, table, natural-language predicate,
optional SQL filter, `limit`). Q1–Q5 of the paper are `yelp_delay`, `yelp_hygiene`,
`yelp_staff_praise`, `amazon_defect` and `amazon_gift`.

### 5.6 Collect evaluation traces

A trace scores every candidate tuple on all three tiers and also stores the keep masks of the
LLM-proposed rules, so that every method and setting can then be replayed offline.

> **Note.** Here the cloud labels *every* tuple, including HIGH and MEDIUM ones. This is only
> done to obtain an evaluation oracle on public data whose personal data is synthetic. The
> online system (`run_live.py`) never does this. Do not run trace collection on real
> personal data.

The device and edge scores must be computed on the Jetson; the cloud labels can be fetched from
any machine, so we fetch them on the workstation in parallel and merge the caches:

1. Cloud labels into a separate cache [Edge] (about 6–9 h per 30,000-tuple query at 20,000 TPM;
   about 5–8 CNY per query at the prices in `default.yaml`):
   ```bash
   sed 's#^cache: .*#cache: results/cache_cloud.sqlite#' configs/default.yaml > configs/prefetch.yaml
   python scripts/collect_trace.py --config configs/prefetch.yaml --prefetch cloud
   scp results/cache_cloud.sqlite nvidia@<JETSON_IP>:~/ptsq/results/
   ```
   Alternatively, `scripts/batch_label.py` submits the labels through the provider's batch API
   (cheaper, no TPM limit) and writes them into the same cache.
2. Device and edge scores, then the traces [Device], inside tmux:
   ```bash
   python scripts/collect_trace.py --prefetch device edge
   python scripts/merge_cache.py results/cache_cloud.sqlite results/cache.sqlite
   python scripts/collect_trace.py --out results/traces          # only cache hits now, writes the traces
   ```
   This writes `results/traces/<qid>.csv` (columns: `privacy`, `tokens`, `s_D`/`y_D`/`t_D`,
   `s_E`/`y_E`/`t_E`, `y_C`/`t_C`/`money_C`, `rule_*`) and `<qid>.rules.json`. Every command can
   be interrupted and re-run: answers already paid for are taken from the cache.
3. Texts for the learned risk and prefilter (same candidates as the traces):
   ```bash
   python scripts/export_texts.py --out results/feat              # results/feat/<qid>_text.csv.gz
   ```
4. Copy `results/traces/`, `results/feat/` and `results/profiles.json` to the workstation and run
   Section 2 with `--traces results/traces`.

### 5.7 End-to-end live runs [Device]

Live runs execute PTSQ online: the Jetson compiles rules with the cloud, certifies each stratum
on a sample labelled by its permitted reference, and streams all tuples through the three tiers
under the shield. They measure real latency and money and are compared with the replay
prediction for the same query (Fig. 10(d)).

1. Make sure vLLM is running (Section 5.3) and the cloud key is set.
2. Pre-compute the TF-IDF features in a separate process (it needs about 1 GB of memory, which
   does not fit next to the device model):
   ```bash
   python scripts/precompute_features.py --config configs/live.yaml --only yelp_delay amazon_defect
   ```
3. Run (about 1.3–1.5 h and 1.1–1.2 CNY per 30,000-tuple query in our setup):
   ```bash
   python scripts/run_live.py --config configs/live.yaml --only yelp_delay amazon_defect \
     --seeds 1 --router none --out results/live.csv
   ```
   `--router none` uses the fixed router evaluated in the paper. Do not pass `--use-cache` for
   measurements: it reuses cached answers and is only a fast functional test.
4. Check the output: `shield_blocked` must be 0; `latency_s`, `money`, `n_device`, `n_edge` and
   `n_cloud` give the measured cost, `plans` the certified plan of each stratum. Copy
   `results/live.csv` to the workstation and pass it to `make_figures.py --live`.

### 5.8 Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Jetson cannot reach `http://<EDGE_IP>:8000` | WSL2 not in mirrored mode, firewall rule missing, or the request goes through a proxy (add the IP to `no_proxy`) |
| `torch.cuda.is_available()` is `False` on the Jetson | a PyPI CPU wheel replaced NVIDIA's; reinstall the NVIDIA wheel, then `pip install -e ".[device,jetson]"` |
| CUDA out of memory on the Jetson | lower `tiers.device.max_batch_tokens` (2048 in `live.yaml`) and run `precompute_features.py` separately |
| Cloud HTTP 429 or 503 | lower `tpm_limit`, `rpm_limit` or `max_concurrency` of the cloud tier; retries back off automatically |
| `s_E` is always 0 or 1 | the edge server does not return log-probabilities; serve the model with vLLM as above (`score_mode: logprobs`) |
