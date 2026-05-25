"""
Verify a headless trace-guided replay against its source simulation and perf log.

Example:
  python legacy_replay_benchmark/verify_headless_replay.py \
    --trace traces/trace_test_headless_trace_1_500.jsonl \
    --replay-sim replay_test_headless_trace_1_500
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE = ROOT / "environment" / "frontend_server" / "storage"


class VerifyError(Exception):
  pass


def load_json(path):
  with open(path, "r", encoding="utf-8") as infile:
    return json.load(infile)


def load_jsonl(path):
  events = []
  with open(path, "r", encoding="utf-8") as infile:
    for line_no, line in enumerate(infile, start=1):
      if not line.strip():
        continue
      try:
        events.append(json.loads(line))
      except json.JSONDecodeError as exc:
        raise VerifyError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
  return events


def canonical(value):
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def count_by_type(events):
  return Counter(event.get("type") for event in events)


def first_event(events, event_type):
  for event in events:
    if event.get("type") == event_type:
      return event
  return None


def default_perf_path(replay_sim):
  safe_sim = "".join(
      char if char.isalnum() or char in ("-", "_") else "_"
      for char in replay_sim
  )
  return Path("perf") / f"headless_replay_perf_{safe_sim}.jsonl"


def latest_default_perf_path(replay_sim):
  base = default_perf_path(replay_sim)
  candidates = [path for path in base.parent.glob(f"{base.stem}*.jsonl") if path.is_file()]
  if not candidates:
    return base
  return max(candidates, key=lambda path: path.stat().st_mtime)


def movement_files(sim_dir):
  movement_dir = sim_dir / "movement"
  if not movement_dir.exists():
    return {}
  files = {}
  for path in movement_dir.glob("*.json"):
    try:
      files[int(path.stem)] = path
    except ValueError:
      continue
  return dict(sorted(files.items()))


def compare_movement(source_sim_dir, replay_sim_dir, trace_events):
  source_files = movement_files(source_sim_dir)
  replay_files = movement_files(replay_sim_dir)
  trace_steps = [
      event.get("step")
      for event in trace_events
      if event.get("type") == "movement_commit"
  ]

  mismatches = []
  missing_source = []
  missing_replay = []

  for step in trace_steps:
    if step not in source_files:
      missing_source.append(step)
      continue
    if step not in replay_files:
      missing_replay.append(step)
      continue
    source = load_json(source_files[step])
    replay = load_json(replay_files[step])
    if canonical(source) != canonical(replay):
      mismatches.append(step)

  return {
      "trace_steps": len(trace_steps),
      "source_count": len(source_files),
      "replay_count": len(replay_files),
      "missing_source": missing_source,
      "missing_replay": missing_replay,
      "mismatches": mismatches,
  }


def compare_meta(source_sim_dir, replay_sim_dir):
  source_meta_path = source_sim_dir / "reverie" / "meta.json"
  replay_meta_path = replay_sim_dir / "reverie" / "meta.json"
  if not source_meta_path.exists() or not replay_meta_path.exists():
    return {
        "status": "missing",
        "source_exists": source_meta_path.exists(),
        "replay_exists": replay_meta_path.exists(),
  }

  source_meta = load_json(source_meta_path)
  replay_meta = load_json(replay_meta_path)
  keys = ["fork_sim_code", "start_date", "curr_time", "sec_per_step", "maze_name", "persona_names", "step"]
  diffs = {}
  for key in keys:
    if source_meta.get(key) != replay_meta.get(key):
      diffs[key] = {
          "source": source_meta.get(key),
          "replay": replay_meta.get(key),
      }
  return {
      "status": "match" if not diffs else "diff",
      "diffs": diffs,
      "source": source_meta,
      "replay": replay_meta,
  }


def trace_summary(events):
  counts = count_by_type(events)
  missing_event_id = sum(1 for event in events if not event.get("event_id"))
  llm_groups = defaultdict(int)
  embedding_groups = defaultdict(int)
  nonempty_retrieval = 0
  movement_chat = 0
  memory_kinds = Counter()
  random_fns = Counter()
  prompt_templates = Counter()
  statuses = Counter()

  for event in events:
    event_type = event.get("type")
    if event_type in ("llm_request", "llm_response"):
      llm_groups[event.get("event_id")] += 1
      if event_type == "llm_response":
        statuses[f"llm:{event.get('status', 'missing')}"] += 1
    elif event_type in ("embedding_request", "embedding_response"):
      embedding_groups[event.get("event_id")] += 1
      if event_type == "embedding_response":
        statuses[f"embedding:{event.get('status', 'missing')}"] += 1
    elif event_type == "retrieval_result":
      if event.get("focal_points"):
        nonempty_retrieval += 1
    elif event_type == "movement_commit":
      for payload in event.get("movement", {}).get("persona", {}).values():
        if payload.get("chat") is not None:
          movement_chat += 1
    elif event_type == "memory_add":
      memory_kinds[event.get("memory_kind")] += 1
    elif event_type == "random_result":
      random_fns[event.get("fn")] += 1
    elif event_type == "prompt_built":
      prompt_templates[event.get("prompt_template")] += 1

  bad_llm_pairs = {
      event_id: count
      for event_id, count in llm_groups.items()
      if event_id and count != 2
  }
  bad_embedding_pairs = {
      event_id: count
      for event_id, count in embedding_groups.items()
      if event_id and count != 2
  }

  return {
      "event_count": len(events),
      "counts": counts,
      "missing_event_id": missing_event_id,
      "bad_llm_pairs": bad_llm_pairs,
      "bad_embedding_pairs": bad_embedding_pairs,
      "nonempty_retrieval": nonempty_retrieval,
      "movement_chat": movement_chat,
      "memory_kinds": memory_kinds,
      "random_fns": random_fns,
      "prompt_templates": prompt_templates,
      "statuses": statuses,
  }


def perf_summary(perf_path):
  if not perf_path or not Path(perf_path).exists():
    return {"exists": False}
  events = load_jsonl(perf_path)
  counts = count_by_type(events)
  statuses = Counter(event.get("status") for event in events)
  total_ms = defaultdict(float)
  max_ms = defaultdict(float)
  errors = []
  for event in events:
    event_type = event.get("type")
    latency = event.get("latency_ms")
    if isinstance(latency, (int, float)):
      total_ms[event_type] += latency
      max_ms[event_type] = max(max_ms[event_type], latency)
    if event.get("status") == "error" or event_type == "replay_error":
      errors.append(event)
  return {
      "exists": True,
      "path": str(perf_path),
      "event_count": len(events),
      "counts": counts,
      "statuses": statuses,
      "total_ms": dict(total_ms),
      "max_ms": dict(max_ms),
      "errors": errors,
  }


def print_counter(title, counter, limit=None):
  print(title)
  items = counter.most_common(limit) if hasattr(counter, "most_common") else list(counter.items())
  if not items:
    print("  none")
    return
  for key, count in items:
    print(f"  {key}: {count}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--trace", required=True)
  parser.add_argument("--source-sim", help="Source simulation name. Defaults to trace simulation_init.sim_code.")
  parser.add_argument("--replay-sim", required=True)
  parser.add_argument("--storage", default=str(DEFAULT_STORAGE))
  parser.add_argument("--perf", help="Replay perf JSONL. Defaults to the latest perf/headless_replay_perf_<replay-sim>*.jsonl.")
  parser.add_argument("--allow-perf-errors", action="store_true")
  args = parser.parse_args()

  trace_path = Path(args.trace)
  perf_path = Path(args.perf) if args.perf else latest_default_perf_path(args.replay_sim)
  storage = Path(args.storage)

  events = load_jsonl(trace_path)
  init = first_event(events, "simulation_init")
  if not init and not args.source_sim:
    raise VerifyError("Trace has no simulation_init; pass --source-sim.")
  source_sim = args.source_sim or init.get("sim_code")

  source_sim_dir = storage / source_sim
  replay_sim_dir = storage / args.replay_sim
  if not source_sim_dir.exists():
    raise VerifyError(f"Source simulation not found: {source_sim_dir}")
  if not replay_sim_dir.exists():
    raise VerifyError(f"Replay simulation not found: {replay_sim_dir}")

  trace = trace_summary(events)
  movement = compare_movement(source_sim_dir, replay_sim_dir, events)
  meta = compare_meta(source_sim_dir, replay_sim_dir)
  perf = perf_summary(perf_path)

  failures = []
  if trace["missing_event_id"]:
    failures.append(f"trace missing event_id: {trace['missing_event_id']}")
  if trace["bad_llm_pairs"]:
    failures.append(f"bad llm request/response pairs: {len(trace['bad_llm_pairs'])}")
  if trace["bad_embedding_pairs"]:
    failures.append(f"bad embedding request/response pairs: {len(trace['bad_embedding_pairs'])}")
  if movement["missing_source"]:
    failures.append(f"missing source movement files: {len(movement['missing_source'])}")
  if movement["missing_replay"]:
    failures.append(f"missing replay movement files: {len(movement['missing_replay'])}")
  if movement["mismatches"]:
    failures.append(f"movement mismatches: {len(movement['mismatches'])}")
  if meta["status"] != "match":
    failures.append(f"meta mismatch/status: {meta['status']}")
  if perf["exists"] and perf["errors"] and not args.allow_perf_errors:
    failures.append(f"perf errors: {len(perf['errors'])}")

  print("Headless Replay Verification")
  print(f"  trace: {trace_path}")
  print(f"  source_sim: {source_sim}")
  print(f"  replay_sim: {args.replay_sim}")
  print(f"  storage: {storage}")
  print("")

  print("Trace")
  print(f"  events: {trace['event_count']}")
  print(f"  missing_event_id: {trace['missing_event_id']}")
  print(f"  bad_llm_pairs: {len(trace['bad_llm_pairs'])}")
  print(f"  bad_embedding_pairs: {len(trace['bad_embedding_pairs'])}")
  print(f"  nonempty_retrieval: {trace['nonempty_retrieval']}")
  print(f"  movement_chat: {trace['movement_chat']}")
  print_counter("  event_counts", trace["counts"])
  print_counter("  memory_kinds", trace["memory_kinds"])
  print_counter("  random_fns", trace["random_fns"])
  print_counter("  statuses", trace["statuses"])
  print_counter("  top_prompt_templates", trace["prompt_templates"], limit=10)
  print("")

  print("Movement")
  print(f"  trace_steps: {movement['trace_steps']}")
  print(f"  source_count: {movement['source_count']}")
  print(f"  replay_count: {movement['replay_count']}")
  print(f"  missing_source: {len(movement['missing_source'])}")
  print(f"  missing_replay: {len(movement['missing_replay'])}")
  print(f"  mismatches: {len(movement['mismatches'])}")
  if movement["mismatches"]:
    print(f"  first_mismatches: {movement['mismatches'][:10]}")
  print("")

  print("Meta")
  print(f"  status: {meta['status']}")
  if meta["status"] == "diff":
    print(f"  diffs: {json.dumps(meta['diffs'], ensure_ascii=True)}")
  print("")

  print("Perf")
  if not perf["exists"]:
    print(f"  missing: {perf_path}")
  else:
    print(f"  path: {perf['path']}")
    print(f"  events: {perf['event_count']}")
    print_counter("  counts", perf["counts"])
    print_counter("  statuses", perf["statuses"])
    for event_type, total in sorted(perf["total_ms"].items()):
      print(f"  total_ms.{event_type}: {total:.3f}")
    for event_type, max_value in sorted(perf["max_ms"].items()):
      print(f"  max_ms.{event_type}: {max_value:.3f}")
    print(f"  errors: {len(perf['errors'])}")
    if perf["errors"]:
      first_error = perf["errors"][0]
      print(f"  first_error: {json.dumps(first_error, ensure_ascii=True)[:1000]}")
  print("")

  if failures:
    print("FAIL")
    for failure in failures:
      print(f"  - {failure}")
    return 1

  print("PASS")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except VerifyError as exc:
    print(f"VERIFY ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)
