"""Export a compact LLM-focused trace from a full GA trace JSONL."""
import argparse
import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(SCRIPT_DIR, "runs")
KEEP_TYPES = {
    "trace_session_start",
    "simulation_init",
    "run_start",
    "run_end",
    "prompt_built",
    "llm_request",
    "llm_response",
    "prompt_result",
}


def _resolve_path(path, base_dir=SCRIPT_DIR):
  if not path:
    return None
  if os.path.exists(path):
    return os.path.abspath(path)
  candidate = os.path.join(base_dir, path)
  if os.path.exists(candidate):
    return os.path.abspath(candidate)
  return os.path.abspath(path)


def _infer_trace_path(trace_or_sim):
  resolved = _resolve_path(trace_or_sim)
  looks_like_path = (
      os.path.exists(resolved)
      or trace_or_sim.endswith(".jsonl")
      or os.path.sep in trace_or_sim
      or "/" in trace_or_sim
      or "\\" in trace_or_sim
  )
  if looks_like_path:
    return resolved, None

  sim = trace_or_sim
  candidates = [
      os.path.join(RUN_ROOT, sim, "trace.jsonl"),
      os.path.join(SCRIPT_DIR, "traces", f"trace_{sim}.jsonl"),
      os.path.join(SCRIPT_DIR, "traces", f"{sim}.jsonl"),
  ]
  for candidate in candidates:
    if os.path.exists(candidate):
      return os.path.abspath(candidate), sim
  raise SystemExit(
      f"Could not infer trace for sim '{sim}':\n  " + "\n  ".join(candidates)
  )


def _default_output(trace_path, sim):
  if sim:
    return os.path.join(RUN_ROOT, sim, "llm_trace.jsonl")
  trace_dir = os.path.dirname(os.path.abspath(trace_path))
  base = os.path.splitext(os.path.basename(trace_path))[0]
  return os.path.join(trace_dir, f"{base}_llm.jsonl")


def export_llm_trace(trace_path, output_path, keep_prompt_result=True):
  keep_types = set(KEEP_TYPES)
  if not keep_prompt_result:
    keep_types.discard("prompt_result")

  total = 0
  kept = 0
  type_counts = {}
  os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
  with open(trace_path, "r", encoding="utf-8") as infile, open(
      output_path,
      "w",
      encoding="utf-8",
      buffering=1,
  ) as outfile:
    for line in infile:
      if not line.strip():
        continue
      total += 1
      record = json.loads(line)
      record_type = record.get("type")
      if record_type not in keep_types:
        continue
      kept += 1
      type_counts[record_type] = type_counts.get(record_type, 0) + 1
      outfile.write(json.dumps(record, ensure_ascii=True) + "\n")

  return {
      "total_records": total,
      "kept_records": kept,
      "type_counts": dict(sorted(type_counts.items())),
      "input_bytes": os.path.getsize(trace_path),
      "output_bytes": os.path.getsize(output_path),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("trace_or_sim", help="Sim name or path to full trace JSONL.")
  parser.add_argument("--output", help="Defaults to runs/<sim>/llm_trace.jsonl.")
  parser.add_argument(
      "--no-prompt-result",
      action="store_true",
      help="Drop prompt_result records. run_grouped_trace_llm.py does not require them.",
  )
  args = parser.parse_args()

  trace_path, sim = _infer_trace_path(args.trace_or_sim)
  output_path = os.path.abspath(args.output or _default_output(trace_path, sim))
  summary = export_llm_trace(
      trace_path,
      output_path,
      keep_prompt_result=not args.no_prompt_result,
  )
  input_mb = summary["input_bytes"] / 1024 / 1024
  output_mb = summary["output_bytes"] / 1024 / 1024
  ratio = (
      summary["output_bytes"] / summary["input_bytes"]
      if summary["input_bytes"]
      else 0.0
  )
  print(f"[llm trace] input: {trace_path}")
  print(f"[llm trace] output: {output_path}")
  print(
      f"[llm trace] records: {summary['kept_records']}/{summary['total_records']} "
      f"bytes: {output_mb:.2f}MB/{input_mb:.2f}MB ({ratio:.1%})"
  )
  print(f"[llm trace] types: {json.dumps(summary['type_counts'], ensure_ascii=True)}")


if __name__ == "__main__":
  main()

