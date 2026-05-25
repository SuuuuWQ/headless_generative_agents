"""
Build conservative interaction groups from a position index.

This implements the Level-2 grouping design:
  - Reconstruct each agent's effective percepts with GA-style perception:
    square vision range, same-arena filtering, event dedupe, distance ranking,
    and attention bandwidth.
  - Add an edge if one agent can perceive another agent.
  - Add an edge if two agents perceive the same object event.
  - Groups are connected components of that conflict graph.

The default output keeps only scheduler-facing step/groups records. Use
--include-percepts to also write edge explanations, per-step stats, and
per-agent percepts. This is intended for trace-driven LLM benchmark scheduling,
not for mutating or re-running GA simulation state.
"""
import argparse
import json
import math
import os
from collections import defaultdict

from maze import Maze
from utils import fs_storage


DEFAULT_VISION_R = 4
DEFAULT_ATT_BANDWIDTH = 3


def _load_json(path):
  with open(path, "r", encoding="utf-8") as infile:
    return json.load(infile)


def _load_jsonl(path):
  with open(path, "r", encoding="utf-8") as infile:
    for line in infile:
      line = line.strip()
      if line:
        yield json.loads(line)


def _write_jsonl(path, records):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8", buffering=1) as outfile:
    for record in records:
      outfile.write(json.dumps(record, ensure_ascii=True) + "\n")


def _safe_int(value, default=None):
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _agent_params(storage_root, sim, agents):
  params = {}
  for agent in agents:
    scratch_path = os.path.join(
        storage_root,
        sim,
        "personas",
        agent,
        "bootstrap_memory",
        "scratch.json",
    )
    vision_r = DEFAULT_VISION_R
    att_bandwidth = DEFAULT_ATT_BANDWIDTH
    if os.path.exists(scratch_path):
      scratch = _load_json(scratch_path)
      vision_r = _safe_int(scratch.get("vision_r"), vision_r)
      att_bandwidth = _safe_int(scratch.get("att_bandwidth"), att_bandwidth)
    params[agent] = {
        "vision_r": vision_r,
        "att_bandwidth": att_bandwidth,
        "scratch_path": os.path.abspath(scratch_path),
    }
  return params


def _load_position_index(path):
  meta = None
  steps = []
  for record in _load_jsonl(path):
    record_type = record.get("type")
    if record_type == "position_index_meta":
      meta = record
    elif record_type == "position_step":
      steps.append(record)
  if meta is None:
    meta = {}
  return meta, steps


def _agent_tile(agent_record):
  x = _safe_int(agent_record.get("x"))
  y = _safe_int(agent_record.get("y"))
  if x is None or y is None:
    return None
  return (x, y)


def _event_id(event):
  subject, predicate, obj, _description = event
  if ":" not in subject:
    return f"agent:{subject}"
  if predicate is None and obj is None:
    return f"object:{subject}"
  return f"object:{subject}|{predicate}|{obj}"


def _event_kind(event_id):
  return event_id.split(":", 1)[0]


def _event_subject(event_id):
  body = event_id.split(":", 1)[1]
  return body.split("|", 1)[0]


def _tile_distance(a, b):
  return math.dist([a[0], a[1]], [b[0], b[1]])


def _collect_object_candidates(agent_tile, agent_arena, maze, vision_r):
  candidates = []
  seen = set()
  for tile in maze.get_nearby_tiles(agent_tile, vision_r):
    try:
      if maze.get_tile_path(tile, "arena") != agent_arena:
        continue
      tile_details = maze.access_tile(tile)
    except Exception:
      continue
    for event in tile_details.get("events", set()):
      event_id = _event_id(event)
      if event_id in seen:
        continue
      seen.add(event_id)
      candidates.append(
          {
              "id": event_id,
              "kind": _event_kind(event_id),
              "subject": _event_subject(event_id),
              "tile": [tile[0], tile[1]],
              "dist": _tile_distance(agent_tile, tile),
          }
      )
  return candidates


def _collect_agent_candidates(observer, agent_tile, agent_arena, agents, maze, vision_r):
  candidates = []
  for target, target_record in agents.items():
    target_tile = _agent_tile(target_record)
    if target_tile is None:
      continue
    try:
      if maze.get_tile_path(target_tile, "arena") != agent_arena:
        continue
    except Exception:
      continue
    if target_tile not in maze.get_nearby_tiles(agent_tile, vision_r):
      continue
    event_id = f"agent:{target}"
    candidates.append(
        {
            "id": event_id,
            "kind": "agent",
            "subject": target,
            "tile": [target_tile[0], target_tile[1]],
            "dist": _tile_distance(agent_tile, target_tile),
            "self": target == observer,
        }
    )
  return candidates


def _effective_percepts(agent, agent_record, agents, maze, params):
  agent_tile = _agent_tile(agent_record)
  if agent_tile is None:
    return []
  vision_r = params[agent]["vision_r"]
  att_bandwidth = params[agent]["att_bandwidth"]
  if att_bandwidth <= 0:
    return []

  agent_arena = agent_record.get("arena_path")
  if not agent_arena:
    try:
      agent_arena = maze.get_tile_path(agent_tile, "arena")
    except Exception:
      return []

  candidates = []
  candidates.extend(
      _collect_object_candidates(agent_tile, agent_arena, maze, vision_r)
  )
  candidates.extend(
      _collect_agent_candidates(agent, agent_tile, agent_arena, agents, maze, vision_r)
  )

  deduped = {}
  for candidate in candidates:
    current = deduped.get(candidate["id"])
    if current is None or candidate["dist"] < current["dist"]:
      deduped[candidate["id"]] = candidate

  ranked = sorted(
      deduped.values(),
      key=lambda item: (item["dist"], item["kind"], item["subject"]),
  )
  if len(ranked) <= att_bandwidth:
    return ranked

  # GA keeps the first att_bandwidth after sorting by distance. We include all
  # events tied at the cutoff distance to avoid false negatives from set order.
  cutoff_dist = ranked[att_bandwidth - 1]["dist"]
  return [item for item in ranked if item["dist"] <= cutoff_dist]


def _add_edge(edges, a, b, reason, detail=None):
  if a == b:
    return
  left, right = sorted([a, b])
  key = (left, right)
  edge = edges.setdefault(
      key,
      {
          "a": left,
          "b": right,
          "reasons": [],
          "objects": [],
          "directions": [],
      },
  )
  if reason not in edge["reasons"]:
    edge["reasons"].append(reason)
  if reason == "shared_object" and detail and detail not in edge["objects"]:
    edge["objects"].append(detail)
  if reason == "perceives_agent" and detail and detail not in edge["directions"]:
    edge["directions"].append(detail)


def _connected_components(agent_names, edges):
  parent = {agent: agent for agent in agent_names}

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(a, b):
    ra = find(a)
    rb = find(b)
    if ra != rb:
      parent[rb] = ra

  for a, b in edges:
    union(a, b)

  groups = defaultdict(list)
  for agent in agent_names:
    groups[find(agent)].append(agent)
  return [sorted(group) for group in groups.values()]


def _group_step(step_record, maze, params, include_percepts):
  agents = step_record.get("agents", {})
  agent_names = sorted(agents)
  percepts = {}
  for agent in agent_names:
    percepts[agent] = _effective_percepts(
        agent,
        agents[agent],
        agents,
        maze,
        params,
    )

  edges = {}

  for observer, items in percepts.items():
    for item in items:
      if item["kind"] == "agent":
        target = item["subject"]
        if target != observer and target in agents:
          _add_edge(
              edges,
              observer,
              target,
              "perceives_agent",
              {"observer": observer, "target": target},
          )

  object_to_agents = defaultdict(list)
  object_subject = {}
  for agent, items in percepts.items():
    for item in items:
      if item["kind"] == "object":
        object_to_agents[item["id"]].append(agent)
        object_subject[item["id"]] = item["subject"]

  for object_id, seen_agents in object_to_agents.items():
    seen_agents = sorted(set(seen_agents))
    if len(seen_agents) < 2:
      continue
    for i, a in enumerate(seen_agents):
      for b in seen_agents[i + 1:]:
        _add_edge(
            edges,
            a,
            b,
            "shared_object",
            object_subject[object_id],
        )

  groups = _connected_components(agent_names, edges.keys())
  result = {
      "type": "group_step",
      "step": step_record.get("step"),
      "groups": groups,
  }
  if include_percepts:
    result["edges"] = sorted(edges.values(), key=lambda item: (item["a"], item["b"]))
    result["stats"] = {
        "agent_count": len(agent_names),
        "group_count": len(groups),
        "edge_count": len(edges),
    }
  if include_percepts:
    result["percepts"] = {
        agent: [
            {
                "id": item["id"],
                "kind": item["kind"],
                "subject": item["subject"],
                "tile": item["tile"],
                "dist": item["dist"],
            }
            for item in items
        ]
        for agent, items in percepts.items()
    }
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--position-index", required=True)
  parser.add_argument("--sim", help="Simulation name. Defaults to position-index meta.")
  parser.add_argument("--storage-root", default=fs_storage)
  parser.add_argument("--maze", help="Maze name. Defaults to position-index meta.")
  parser.add_argument(
      "--output",
      help="Defaults to <position-index folder>/group_index.jsonl.",
  )
  parser.add_argument(
      "--include-percepts",
      action="store_true",
      help="Include edge explanations, per-step stats, and each agent's effective percept set.",
  )
  args = parser.parse_args()

  meta, steps = _load_position_index(args.position_index)
  if not steps:
    raise ValueError(f"No position_step records found in {args.position_index}")

  sim = args.sim or meta.get("sim")
  if not sim:
    parser.error("Could not infer simulation name. Provide --sim.")
  maze_name = args.maze or meta.get("maze")
  if not maze_name:
    parser.error("Could not infer maze name. Provide --maze.")

  first_agents = sorted(steps[0].get("agents", {}))
  params = _agent_params(os.path.abspath(args.storage_root), sim, first_agents)
  maze = Maze(maze_name)
  output_path = args.output or os.path.join(
      os.path.dirname(os.path.abspath(args.position_index)),
      "group_index.jsonl",
  )

  records = [
      {
          "type": "group_index_meta",
          "position_index": os.path.abspath(args.position_index),
          "sim": sim,
          "storage_root": os.path.abspath(args.storage_root),
          "maze": maze_name,
          "start_step": meta.get("start_step"),
          "end_step": meta.get("end_step"),
          "agent_params": params,
          "rules": {
              "vision": "GA square vision_r",
              "arena_filter": "same arena_path",
              "dedupe": "event id",
              "ranking": "distance, tie-inclusive at att_bandwidth cutoff",
              "edges": ["perceives_agent", "shared_object"],
              "retention": "not applied",
          },
      }
  ]
  for step_record in steps:
    records.append(
        _group_step(step_record, maze, params, args.include_percepts)
    )

  _write_jsonl(output_path, records)
  print(f"[groups] wrote {output_path}")
  print(f"[groups] steps={len(steps)} agents={len(first_agents)}")


if __name__ == "__main__":
  main()
