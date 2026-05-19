"""
Run the headless GA directly while recording both trace and direct-run perf.

Example:
  TRACE_RECORD_PREFIX_SNAPSHOTS=true \
  TRACE_RECORD_PROMPT_RESULT=true \
  python headless_direct_trace_perf_runner.py \
    --fork July1_the_ville_isabella_maria_klaus-step-3-14 \
    --sim direct_trace_perf_100_a \
    --steps 100
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

import sglang_openai_patch

import headless_direct_perf_runner as direct_perf
import headless_trace_runner as trace_runner


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(SCRIPT_DIR, "direct_run_perf_runs")


def _configure_seed(seed):
  if seed is None:
    return None
  seed = int(seed)
  random.seed(seed)
  try:
    import numpy

    numpy.random.seed(seed)
  except Exception:
    pass
  return seed


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--fork", help="Simulation folder to fork from.")
  parser.add_argument("--sim", help="New simulation folder name.")
  parser.add_argument("--steps", type=int)
  parser.add_argument(
      "--run-dir",
      help="Defaults to <this script dir>/direct_run_perf_runs/<sim>.",
  )
  parser.add_argument("--perf", help="Defaults to <run-dir>/perf.jsonl.")
  parser.add_argument(
      "--seed",
      type=int,
      default=os.environ.get("GA_RANDOM_SEED"),
      help="Seed Python random and NumPy. Can also be set with GA_RANDOM_SEED.",
  )
  parser.add_argument("--no-save", action="store_true")
  args = parser.parse_args()

  fork = args.fork or input("Enter the name of the forked simulation: ").strip()
  sim = args.sim or input("Enter the name of the new simulation: ").strip()
  run_dir = args.run_dir or os.path.join(RUN_ROOT, sim)
  perf_path = args.perf or os.path.join(run_dir, "perf.jsonl")
  os.makedirs(run_dir, exist_ok=True)
  seed = _configure_seed(args.seed)
  hash_seed = os.environ.get("PYTHONHASHSEED")

  with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as outfile:
    json.dump(
        {
            "mode": "direct_headless_trace_perf",
            "fork": fork,
            "sim": sim,
            "steps": args.steps,
            "run_dir": os.path.abspath(run_dir),
            "perf": os.path.abspath(perf_path),
            "trace_dir": trace_runner.TRACE_DIR,
            "trace_file": trace_runner.TRACE_FILE,
            "seed": seed,
            "pythonhashseed": hash_seed,
            "sglang_force_temperature": os.environ.get("SGLANG_FORCE_TEMPERATURE"),
            "sglang_force_top_p": os.environ.get("SGLANG_FORCE_TOP_P"),
            "sglang_request_seed": os.environ.get("SGLANG_REQUEST_SEED"),
        },
        outfile,
        indent=2,
    )

  direct_perf.DIRECT = direct_perf.DirectPerfController(perf_path)

  sglang_openai_patch.HEADLESS_ENVIRONMENT = False
  sglang_openai_patch._install_logging()
  sglang_openai_patch._install_default_utils_if_missing()
  sglang_openai_patch._install_fast_fork_copy()
  sglang_openai_patch._install_reverie_server_init_hook()

  trace_runner._install_import_trace_hooks()
  trace_runner._install_class_trace_hooks()
  trace_runner._install_movement_trace_hook()
  trace_runner._install_openai_trace_hooks()
  trace_runner._install_random_trace_hooks()
  direct_perf._install_prompt_perf_hooks()

  trace_runner.TRACE.emit(
      "trace_session_start",
      event_id=trace_runner.TRACE.global_event_id("trace_session", label="start"),
      trace_file=trace_runner.TRACE_FILE,
      trace_dir=trace_runner.TRACE_DIR,
      argv=sys.argv,
      sglang_model=sglang_openai_patch.SGLANG_MODEL,
      sglang_api_base=sglang_openai_patch.SGLANG_API_BASE,
      sglang_embedding_model=sglang_openai_patch.SGLANG_EMBEDDING_MODEL,
      sglang_embedding_api_base=sglang_openai_patch.SGLANG_EMBEDDING_API_BASE,
      record_full_prompt=trace_runner.TRACE_RECORD_FULL_PROMPT,
      record_embeddings=trace_runner.TRACE_RECORD_EMBEDDINGS,
      seed=seed,
      pythonhashseed=hash_seed,
      sglang_force_temperature=os.environ.get("SGLANG_FORCE_TEMPERATURE"),
      sglang_force_top_p=os.environ.get("SGLANG_FORCE_TOP_P"),
      sglang_request_seed=os.environ.get("SGLANG_REQUEST_SEED"),
  )

  start_time_ns = time.time_ns()
  status = "ok"
  error = None
  try:
    from reverie import ReverieServer

    direct_perf._install_openai_perf_hooks()
    direct_perf._install_class_perf_hooks()

    print(f"[direct trace+perf] fork={fork} sim={sim} steps={args.steps}")
    print(f"[direct trace+perf] run_dir={os.path.abspath(run_dir)}")
    server = ReverieServer(fork, sim)
    direct_perf.DIRECT.set_server(server)

    if args.steps is None:
      server.open_server()
      print(f"[direct trace+perf] interactive session complete: {sim}")
    else:
      server.start_server(args.steps)
      if not args.no_save:
        server.save()
      print(f"[direct trace+perf] complete: {sim}, steps={args.steps}")

    print(f"[direct trace+perf] perf: {os.path.abspath(perf_path)}")
  except Exception:
    status = "error"
    error = traceback.format_exc()
    direct_perf.DIRECT.perf.record(
        type="direct_run_error",
        status="error",
        error=error,
    )
    trace_runner.TRACE.emit(
        "trace_session_error",
        event_id=trace_runner.TRACE.global_event_id("trace_session", label="error"),
        error=error,
    )
    raise
  finally:
    end_time_ns = time.time_ns()
    llm_ms = direct_perf.DIRECT.perf.latency_by_type.get("worker_llm", 0.0)
    embedding_ms = direct_perf.DIRECT.perf.latency_by_type.get("worker_embedding", 0.0)
    wall_ms = (end_time_ns - start_time_ns) / 1_000_000
    direct_perf.DIRECT.perf.record(
        type="direct_run_total",
        status=status,
        error=error,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=wall_ms,
        llm_total_ms=llm_ms,
        embedding_total_ms=embedding_ms,
        non_llm_total_ms=max(0.0, wall_ms - llm_ms),
        overhead_excluding_llm_embedding_ms=max(0.0, wall_ms - llm_ms - embedding_ms),
        event_counts=direct_perf.DIRECT.perf.counts,
        status_counts=direct_perf.DIRECT.perf.status_counts,
    )
    direct_perf.DIRECT.close()
    trace_runner.TRACE.emit(
        "trace_session_end",
        event_id=trace_runner.TRACE.global_event_id("trace_session", label="end"),
    )
    trace_runner.TRACE.close()


if __name__ == "__main__":
  main()
