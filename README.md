# Headless Generative Agents Replay Benchmark

This package is a self-contained headless copy of the Generative Agents backend.
It removes the need for the browser visualization loop while preserving the GA
control flow, persona logic, memory structures, planning, reflection, retrieval,
movement writing, trace recording, trace-guided replay, verification, profiling,
and plotting.

The package includes:

- `headless_genagents/`: headless GA code and replay/profiling tools.
- `environment/frontend_server/storage/July1_the_ville_isabella_maria_klaus-step-3-14/`:
  the recommended fork point for new smoke tests and traces.
- `environment/frontend_server/storage/test_headless_trace_1_500/`: source
  simulation output for the included 500-step trace.
- `environment/frontend_server/storage/replay_test_headless_trace_1_500_l/`: a
  successful replay output for comparison.
- `headless_genagents/traces/trace_test_headless_trace_1_500.jsonl`: included
  replay trace.
- `headless_genagents/replay_runs/replay_test_headless_trace_1_500_l/`:
  included example replay run with `config.json`, `perf.jsonl`,
  `report.json`, `report.md`, and `figures/`.

## Code Map

Most files in `headless_genagents/` are copied from the original
`reverie/backend_server` backend to keep the GA control flow close to the
original system. The main added or modified entry points are:

- `headless_runner.py`: run GA without the browser visualization loop.
- `headless_trace_runner.py`: run headless GA and record a replay trace.
- `headless_replay_runner.py`: replay a trace through the real headless GA
  control flow while timing model calls and using trace values for
  nondeterministic results.
- `verify_headless_replay.py`: check trace integrity, replay output, metadata,
  and performance logs.
- `plot_replay_report.py`: generate profiling figures from a replay report.
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

## Record A Trace

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

## Replay The Included Trace

Run a new replay from the included trace with a new simulation name:

```bash
cd headless_genagents
python headless_replay_runner.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --sim replay_test_headless_trace_1_500_new
```

By default, the replay profiling artifacts are saved under one run folder:

```text
headless_genagents/replay_runs/<replay-sim>/
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
python headless_replay_runner.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --sim replay_test_headless_trace_1_500_new \
  --run-dir replay_runs/replay_test_headless_trace_1_500_new
```

## Report And Figures

Each replay run writes `report.json` and `report.md` automatically. To generate
figures for a replay run:

```bash
python plot_replay_report.py \
  --report replay_runs/replay_test_headless_trace_1_500_new/report.json
```

Figures are written to:

```text
headless_genagents/replay_runs/<replay-sim>/figures
```

If you already have a perf log and only want to regenerate reports:

```bash
python headless_replay_runner.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --sim replay_test_headless_trace_1_500_new \
  --perf replay_runs/replay_test_headless_trace_1_500_new/perf.jsonl \
  --report replay_runs/replay_test_headless_trace_1_500_new/report.json \
  --report-md replay_runs/replay_test_headless_trace_1_500_new/report.md \
  --report-only
```

## Verify Replay Output

To verify a replay run:

```bash
python verify_headless_replay.py \
  --trace traces/trace_test_headless_trace_1_500.jsonl \
  --replay-sim replay_test_headless_trace_1_500_new \
  --perf replay_runs/replay_test_headless_trace_1_500_new/perf.jsonl \
  --allow-perf-errors
```

The package also includes one completed replay run. To verify the included
example:

```bash
python verify_headless_replay.py \
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
