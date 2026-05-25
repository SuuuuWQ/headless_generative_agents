# Headless Generative Agents LLM Benchmark

This package is a self-contained headless copy of the Generative Agents backend.
It removes the need for the browser visualization loop while preserving the GA
control flow, persona logic, memory structures, planning, reflection, retrieval,
movement writing, trace recording, profiling, and plotting. The main benchmark
path records one direct GA run, builds interaction groups from the original GA
perception logic, and then reruns only the trace's LLM requests with grouped
scheduling.

The package includes:

- `headless_genagents/`: headless GA code and benchmark/profiling tools.
- `environment/frontend_server/storage/July1_the_ville_isabella_maria_klaus-step-3-14/`:
  the recommended fork point for new smoke tests and traces.
- `environment/frontend_server/storage/test_headless_trace_1_500/`: source
  simulation output for the included 500-step trace.
- `environment/frontend_server/storage/replay_test_headless_trace_1_500_l/`: a
  successful replay output for comparison.
- `headless_genagents/legacy_replay_benchmark/`: older trace-guided replay
  benchmark scripts kept for reference.
- `headless_genagents/runs/direct_trace_perf_n25_10_l/`: included grouped LLM
  benchmark example with `llm_trace.jsonl`, `position_index.jsonl`, and
  `group_index.jsonl` only. The full `trace.jsonl` is intentionally omitted.

## Code Map

Most files in `headless_genagents/` are copied from the original
`reverie/backend_server` backend to keep the GA control flow close to the
original system. The main added or modified tools are grouped below.

### Run And Record

- `headless_runner.py`: run GA without the browser visualization loop.
- `headless_trace_runner.py`: run headless GA and record a replay trace.
- `headless_direct_trace_perf_runner.py`: run headless GA once and record both
  a full replay trace and direct-run performance logs. This is the recommended
  entry point for new trace collection.
- `headless_direct_perf_runner.py`: run headless GA and record performance logs
  without writing a full replay trace.

### Grouped LLM Benchmark

- `export_position_index.py`: export compact per-step agent positions from
  `environment/<step>.json`.
- `export_llm_trace.py`: export a compact LLM-focused trace from a full trace.
- `build_interaction_groups.py`: build conservative interaction groups using
  GA-style perception rules.
- `run_grouped_trace_llm.py`: rerun only trace LLM requests with step order,
  group-parallel scheduling, group-internal serial execution, and agent-local
  LLM order preserved.
- `plot_group_index.py`: visualize a step's interaction groups and percepts.

### Plot

- `plot_direct_perf.py`: generate direct-run timeline figures.
- `plot_grouped_trace_llm.py`: generate grouped LLM benchmark timelines.

### Legacy Replay Benchmark

- `legacy_replay_benchmark/headless_replay_runner.py`: older trace-guided
  replay path through GA control flow.
- `legacy_replay_benchmark/compare_ga_behavior.py`: compare direct/replay
  artifacts.
- `legacy_replay_benchmark/verify_headless_replay.py`: verify replay output.
- `legacy_replay_benchmark/plot_replay_report.py`: plot replay reports.

### Support

- `sglang_openai_patch.py`: route legacy OpenAI calls to local
  OpenAI-compatible SGLang endpoints.
- `utils.py`: local path and API configuration for this self-contained copy.

## Setup

Use the same Python environment as the original Generative Agents project. The
root `requirements.txt` is included as a starting point.

```bash
pip install -r requirements.txt
```

## Model Server Configuration

This code uses OpenAI-compatible HTTP endpoints. Configure the LLM and embedding
servers in `headless_genagents/sglang_openai_patch.py`, or set environment
variables before running.

The most important fields in `sglang_openai_patch.py` are:

```python
SGLANG_API_KEY = "dummy"
SGLANG_API_BASE = "http://127.0.0.1:1919/v1"
SGLANG_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

SGLANG_EMBEDDING_API_BASE = "http://127.0.0.1:1920/v1"
SGLANG_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
```

Use the same API key that you passed to your SGLang server. If your server was
started without an API key, `dummy` is usually fine.

You can also override most settings from the shell:

```bash
export SGLANG_API_BASE=http://127.0.0.1:1919/v1
export SGLANG_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export SGLANG_EMBEDDING_API_BASE=http://127.0.0.1:1920/v1
export SGLANG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

`SGLANG_API_KEY` is currently a constant in `sglang_openai_patch.py`; edit that
line directly if your server requires a different key.

For lower-variance runs, you can also force common sampling settings:

```bash
export SGLANG_FORCE_TEMPERATURE=0
export SGLANG_FORCE_TOP_P=1
export SGLANG_REQUEST_SEED=1
export GA_RANDOM_SEED=1
export PYTHONHASHSEED=1
```

`SGLANG_FORCE_TEMPERATURE` and `SGLANG_FORCE_TOP_P` override the temperature and
top-p sent by legacy GA prompts. `SGLANG_REQUEST_SEED` is passed through to
OpenAI-compatible servers that support per-request seeds. `GA_RANDOM_SEED`
seeds Python and NumPy inside the direct trace+perf runner.

Example local SGLang launch commands:

```bash
python -m sglang.launch_server \
  --model-path /path/to/llm \
  --host 0.0.0.0 \
  --port 1919 \
  --api-key dummy

python -m sglang.launch_server \
  --model-path /path/to/embedding-model \
  --is-embedding \
  --host 0.0.0.0 \
  --port 1920 \
  --api-key dummy
```

## Local Paths

The default local paths are self-contained:

```text
environment/frontend_server/storage
environment/frontend_server/static_dirs/assets
environment/frontend_server/temp_storage
```

You can override them with:

```bash
export FS_STORAGE=/path/to/storage
export FS_TEMP_STORAGE=/path/to/temp_storage
export MAZE_ASSETS_LOC=/path/to/assets
```

## Smoke Test: Run Headless GA

From this package root:

```bash
cd headless_genagents
python headless_runner.py \
  --fork July1_the_ville_isabella_maria_klaus-step-3-14 \
  --sim headless_smoke \
  --steps 5
```

This should create:

```text
../environment/frontend_server/storage/headless_smoke
```

## Record A Trace With Direct Performance Logs

This is the recommended command for new experiments because it records the full
trace, direct-run performance log, position index, and group index in one run:

```bash
cd headless_genagents
python headless_direct_trace_perf_runner.py \
  --fork July1_the_ville_isabella_maria_klaus-step-3-14 \
  --sim direct_trace_perf_100 \
  --steps 100 \
  --seed 1 \
  --export-indexes
```

This writes:

```text
headless_genagents/runs/direct_trace_perf_100/
  config.json
  trace.jsonl
  llm_trace.jsonl
  perf.jsonl
  position_index.jsonl
  group_index.jsonl
```

Prefix snapshots are optional and disabled by default. Enable them only when you
need to resume or debug from intermediate prefixes:

```bash
export TRACE_RECORD_PREFIX_SNAPSHOTS=true
```

To generate direct-run timeline figures:

```bash
python plot_direct_perf.py \
  --perf runs/direct_trace_perf_100/perf.jsonl
```

To create a smaller trace for sharing or grouped-LLM benchmarking:

```bash
python export_llm_trace.py direct_trace_perf_100
```

This writes:

```text
headless_genagents/runs/direct_trace_perf_100/llm_trace.jsonl
```

## Run The Grouped LLM Benchmark

After recording a direct trace+perf run with `--export-indexes`, rerun only the
trace's LLM requests with interaction-group scheduling:

```bash
python run_grouped_trace_llm.py direct_trace_perf_100 --plot
```

The grouped benchmark reads:

```text
headless_genagents/runs/direct_trace_perf_100/llm_trace.jsonl
headless_genagents/runs/direct_trace_perf_100/group_index.jsonl
```

If `llm_trace.jsonl` is missing, it falls back to `trace.jsonl`.

and writes:

```text
headless_genagents/runs/direct_trace_perf_100/grouped_trace_llm_perf.jsonl
headless_genagents/runs/direct_trace_perf_100/figures/
```

To inspect the grouping for one step:

```bash
python plot_group_index.py direct_trace_perf_100 --step 2880
```

The release includes one compact example run:

```bash
python run_grouped_trace_llm.py direct_trace_perf_n25_10_l --plot
python plot_group_index.py direct_trace_perf_n25_10_l --step 2880
```

This example includes `llm_trace.jsonl`, `position_index.jsonl`, and
`group_index.jsonl`; it does not include the full `trace.jsonl`.

## Record A Trace Only

Interactive mode:

```bash
cd headless_genagents
python headless_trace_runner.py \
  --fork July1_the_ville_isabella_maria_klaus-step-3-14 \
  --sim my_trace_sim
```

Then enter commands:

```text
run 20
run 100
fin
```

One-shot mode:

```bash
python headless_trace_runner.py \
  --fork July1_the_ville_isabella_maria_klaus-step-3-14 \
  --sim my_trace_sim \
  --steps 20
```

The trace is written to:

```text
headless_genagents/traces/trace_my_trace_sim.jsonl
```

## Legacy: Replay The Included Trace

Run a new replay from the included trace with a new simulation name:

```bash
cd headless_genagents
python legacy_replay_benchmark/headless_replay_runner.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --sim replay_test_headless_trace_1_500_new
```

By default, the replay profiling artifacts are saved under one run folder:

```text
headless_genagents/legacy_replay_benchmark/replay_runs/<replay-sim>/
  config.json
  perf.jsonl
  report.json
  report.md
```

This keeps profiling artifacts together. The GA simulation output is still
written to the normal storage folder:

```text
environment/frontend_server/storage/<replay-sim>
```

To choose the run folder explicitly:

```bash
python legacy_replay_benchmark/headless_replay_runner.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --sim replay_test_headless_trace_1_500_new \
  --run-dir legacy_replay_benchmark/replay_runs/replay_test_headless_trace_1_500_new
```

To replay a trace you just recorded:

```bash
python legacy_replay_benchmark/headless_replay_runner.py \
  --trace runs/direct_trace_perf_100/trace.jsonl \
  --sim replay_direct_trace_perf_100
```

## Legacy: Compare Direct Run And Replay

After recording a direct trace+perf run and replaying it, compare behavior and
LLM-call alignment:

```bash
python legacy_replay_benchmark/compare_ga_behavior.py \
  runs/direct_trace_perf_100 \
  replay_runs/replay_direct_trace_perf_100
```

The behavior-level comparison checks agent-round outputs, retrieval/reflection
counts, memory writes, and model-call signatures. The LLM-call comparison is
the most useful view for steps that contain many model calls.

## Legacy: Report And Figures

Each replay run writes `report.json` and `report.md` automatically. To generate
figures for a replay run:

```bash
python legacy_replay_benchmark/plot_replay_report.py \
  --report legacy_replay_benchmark/replay_runs/replay_test_headless_trace_1_500_new/report.json
```

Figures are written to:

```text
headless_genagents/legacy_replay_benchmark/replay_runs/<replay-sim>/figures
```

If you already have a perf log and only want to regenerate reports:

```bash
python legacy_replay_benchmark/headless_replay_runner.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --sim replay_test_headless_trace_1_500_new \
  --perf legacy_replay_benchmark/replay_runs/replay_test_headless_trace_1_500_new/perf.jsonl \
  --report legacy_replay_benchmark/replay_runs/replay_test_headless_trace_1_500_new/report.json \
  --report-md legacy_replay_benchmark/replay_runs/replay_test_headless_trace_1_500_new/report.md \
  --report-only
```

## Legacy: Verify Replay Output

To verify a replay run:

```bash
python legacy_replay_benchmark/verify_headless_replay.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --replay-sim replay_test_headless_trace_1_500_new \
  --perf legacy_replay_benchmark/replay_runs/replay_test_headless_trace_1_500_new/perf.jsonl \
  --allow-perf-errors
```

The package also includes one completed replay run. To verify the included
example:

```bash
python legacy_replay_benchmark/verify_headless_replay.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --replay-sim replay_test_headless_trace_1_500_l \
  --perf replay_runs/replay_test_headless_trace_1_500_l/perf.jsonl \
  --allow-perf-errors
```

## What The Included Report Shows

The included 500-step replay completed successfully. It shows that the workload
is model-bound: LLM plus embedding account for more than 90% of wall time, with
embedding alone around one third of the total. It also shows uneven agent load
and limited speedup from simple same-step agent parallelism, motivating future
work on asynchronous simulation, model request batching, embedding optimization,
and scheduling across cognitive modules.
