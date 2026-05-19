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
original system. The main added or modified tools are grouped below.

### Run And Record

- `headless_runner.py`: run GA without the browser visualization loop.
- `headless_trace_runner.py`: run headless GA and record a replay trace.
- `headless_direct_trace_perf_runner.py`: run headless GA once and record both
  a full replay trace and direct-run performance logs. This is the recommended
  entry point for new trace collection.
- `headless_direct_perf_runner.py`: run headless GA and record performance logs
  without writing a full replay trace.

### Replay And Check

- `headless_replay_runner.py`: replay a trace through the real headless GA
  control flow while timing model calls and using trace values for
  nondeterministic results.
- `compare_ga_behavior.py`: compare direct and replay artifacts at behavior and
  LLM-call level.
- `verify_headless_replay.py`: check trace integrity, replay output, metadata,
  and performance logs.

### Plot

- `plot_direct_perf.py`: generate direct-run timeline figures.
- `plot_replay_report.py`: generate profiling figures from a replay report.

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
trace and a direct-run performance log in one run:

```bash
cd headless_genagents
python headless_direct_trace_perf_runner.py \
  --fork July1_the_ville_isabella_maria_klaus-step-3-14 \
  --sim direct_trace_perf_100 \
  --steps 100 \
  --seed 1
```

This writes:

```text
headless_genagents/traces/trace_direct_trace_perf_100.jsonl
headless_genagents/direct_run_perf_runs/direct_trace_perf_100/
  config.json
  perf.jsonl
```

Prefix snapshots are optional and disabled by default. Enable them only when you
need to resume or debug from intermediate prefixes:

```bash
export TRACE_RECORD_PREFIX_SNAPSHOTS=true
```

To generate direct-run timeline figures:

```bash
python plot_direct_perf.py \
  --perf direct_run_perf_runs/direct_trace_perf_100/perf.jsonl
```

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

To replay a trace you just recorded:

```bash
python headless_replay_runner.py \
  --trace traces/trace_direct_trace_perf_100.jsonl \
  --sim replay_direct_trace_perf_100
```

## Compare Direct Run And Replay

After recording a direct trace+perf run and replaying it, compare behavior and
LLM-call alignment:

```bash
python compare_ga_behavior.py \
  direct_run_perf_runs/direct_trace_perf_100 \
  replay_runs/replay_direct_trace_perf_100
```

The behavior-level comparison checks agent-round outputs, retrieval/reflection
counts, memory writes, and model-call signatures. The LLM-call comparison is
the most useful view for steps that contain many model calls.

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
