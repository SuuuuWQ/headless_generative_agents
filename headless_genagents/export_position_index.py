"""
Export a compact position index from storage/<sim>/environment/*.json.

The environment folder can contain files inherited from the forked simulation.
Use --trace or --start-step/--end-step to restrict the export to the steps that
were actually executed in the current run.

Examples:
  python export_position_index.py \
    --sim direct_trace_perf_100_a \
    --trace runs/direct_trace_perf_100_a/trace.jsonl

  python export_position_index.py \
    --sim direct_trace_perf_100_a \
    --start-step 6137 \
    --end-step 6236
"""
import argparse
import json
import os
import sys

from maze import Maze
from utils import fs_storage


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(SCRIPT_DIR, "runs")


def _safe_int(value, default=None):
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _load_json(path):
  with open(path, "r", encoding="utf-8") as infile:
    return json.load(infile)


def _load_jsonl(path):
  with open(path, "r", encoding="utf-8") as infile:
    for line in infile:
      line = line.strip()
      if not line:
        continue
      yield json.loads(line)


def _infer_range_from_trace(trace_path):
  ranges = []
  sim_code = None
  maze_name = None
  personas = None
  sec_per_step = None
  for event in _load_jsonl(trace_path):
    event_type = event.get("type")
    if event_type == "simulation_init":
      sim_code = event.get("sim_code") or sim_code
      maze_name = event.get("maze_name") or maze_name
      personas = event.get("personas") or personas
      sec_per_step = event.get("sec_per_step") or sec_per_step
    elif event_type == "run_start":
      start_step = _safe_int(event.get("start_step"))
      requested_steps = _safe_int(event.get("requested_steps"))
      if start_step is not None and requested_steps is not None:
        ranges.append((start_step, start_step + requested_steps - 1))

  if not ranges:
    return None
  return {
      "start_step": min(start for start, _ in ranges),
      "end_step": max(end for _, end in ranges),
      "sim_code": sim_code,
      "maze_name": maze_name,
      "personas": personas,
      "sec_per_step": sec_per_step,
      "ranges": ranges,
  }


def _load_meta(sim_dir):
  meta_path = os.path.join(sim_dir, "reverie", "meta.json")
  if os.path.exists(meta_path):
    return _load_json(meta_path)
  return {}


def _infer_maze_name(sim_dir, explicit_maze=None, trace_info=None):
  if explicit_maze:
    return explicit_maze
  if trace_info and trace_info.get("maze_name"):
    return trace_info["maze_name"]
  meta = _load_meta(sim_dir)
  if meta.get("maze_name"):
    return meta["maze_name"]

  env_dir = os.path.join(sim_dir, "environment")
  if os.path.isdir(env_dir):
    for name in sorted(os.listdir(env_dir)):
      if not name.endswith(".json"):
        continue
      env = _load_json(os.path.join(env_dir, name))
      for value in env.values():
        if isinstance(value, dict) and value.get("maze"):
          return value["maze"]
  return None


def _tile_details(maze, tile):
  if maze is None:
    return {}
  try:
    return {
        "arena_path": maze.get_tile_path(tile, "arena"),
    }
  except Exception as exc:
    return {"tile_lookup_error": repr(exc)}


def _step_record(step, env_path, maze):
  env = _load_json(env_path)
  agents = {}
  for agent, loc in sorted(env.items()):
    x = _safe_int(loc.get("x"))
    y = _safe_int(loc.get("y"))
    tile = [x, y] if x is not None and y is not None else None
    record = {
        "x": x,
        "y": y,
    }
    if tile is not None:
      record.update(_tile_details(maze, tuple(tile)))
    agents[agent] = record
  return {
      "type": "position_step",
      "step": step,
      "agents": agents,
  }


def _write_jsonl(path, records):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8", buffering=1) as outfile:
    for record in records:
      outfile.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sim", help="Simulation name under storage.")
  parser.add_argument(
      "--storage-root",
      default=fs_storage,
      help="Defaults to utils.fs_storage.",
  )
  parser.add_argument(
      "--environment-dir",
      help="Environment directory. Overrides --sim/--storage-root.",
  )
  parser.add_argument("--trace", help="Trace JSONL used to infer run step range.")
  parser.add_argument("--start-step", type=int, help="First environment step to export.")
  parser.add_argument("--end-step", type=int, help="Last environment step to export, inclusive.")
  parser.add_argument(
      "--include-final",
      action="store_true",
      help="Also include environment/<end-step + 1>.json.",
  )
  parser.add_argument("--maze", help="Maze name. Defaults to trace/meta/environment value.")
  parser.add_argument(
      "--no-arena",
      action="store_true",
      help="Only export x/y positions; skip maze tile lookup.",
  )
  parser.add_argument(
      "--output",
      help="Defaults to runs/<sim>/position_index.jsonl next to this script.",
  )
  args = parser.parse_args()

  if not args.sim and not args.environment_dir:
    parser.error("Provide --sim or --environment-dir.")

  sim_dir = None
  if args.environment_dir:
    env_dir = os.path.abspath(args.environment_dir)
    sim_dir = os.path.dirname(env_dir)
  else:
    sim_dir = os.path.join(os.path.abspath(args.storage_root), args.sim)
    env_dir = os.path.join(sim_dir, "environment")

  if not os.path.isdir(env_dir):
    raise FileNotFoundError(f"environment directory not found: {env_dir}")

  trace_info = _infer_range_from_trace(args.trace) if args.trace else None
  start_step = args.start_step
  end_step = args.end_step
  if trace_info:
    if start_step is None:
      start_step = trace_info["start_step"]
    if end_step is None:
      end_step = trace_info["end_step"]

  if start_step is None or end_step is None:
    parser.error(
        "Could not determine step range. Provide --trace or both "
        "--start-step and --end-step."
    )
  if end_step < start_step:
    parser.error("--end-step must be >= --start-step.")
  output_end_step = end_step + 1 if args.include_final else end_step

  maze_name = _infer_maze_name(sim_dir, args.maze, trace_info)
  maze = None
  if not args.no_arena:
    if not maze_name:
      parser.error("Could not infer maze name. Provide --maze or use --no-arena.")
    maze = Maze(maze_name)

  output_path = args.output
  if not output_path:
    output_sim = args.sim or os.path.basename(sim_dir)
    output_path = os.path.join(RUN_ROOT, output_sim, "position_index.jsonl")

  records = []
  records.append(
      {
          "type": "position_index_meta",
          "sim": args.sim or os.path.basename(sim_dir),
          "storage_root": os.path.abspath(args.storage_root),
          "environment_dir": os.path.abspath(env_dir),
          "trace": os.path.abspath(args.trace) if args.trace else None,
          "maze": maze_name,
          "start_step": start_step,
          "end_step": end_step,
          "include_final": args.include_final,
          "exported_end_step": output_end_step,
          "trace_ranges": trace_info.get("ranges") if trace_info else None,
      }
  )

  missing = []
  for step in range(start_step, output_end_step + 1):
    env_path = os.path.join(env_dir, f"{step}.json")
    if not os.path.exists(env_path):
      missing.append(step)
      continue
    records.append(_step_record(step, env_path, maze))

  if missing:
    records.append(
        {
            "type": "position_index_warning",
            "message": "missing environment files",
            "missing_steps": missing,
        }
    )

  _write_jsonl(output_path, records)
  exported_steps = sum(1 for record in records if record.get("type") == "position_step")
  print(f"[position index] wrote {output_path}")
  print(f"[position index] exported_steps={exported_steps}")
  print(f"[position index] step_range={start_step}..{output_end_step}")
  if missing:
    print(f"[position index] missing_steps={len(missing)}", file=sys.stderr)


if __name__ == "__main__":
  main()
