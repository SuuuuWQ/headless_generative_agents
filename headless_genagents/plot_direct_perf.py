"""Plot timeline figures for direct headless GA perf logs."""
import argparse
import json
import os
from collections import defaultdict


def load_jsonl(path):
  events = []
  with open(path, "r", encoding="utf-8") as infile:
    for line in infile:
      if line.strip():
        events.append(json.loads(line))
  return events


def ensure_matplotlib():
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  return plt


def subtract_intervals(base_start, base_end, intervals):
  remaining = [(base_start, base_end)]
  for start, end in sorted(intervals):
    next_remaining = []
    for left, right in remaining:
      if end <= left or start >= right:
        next_remaining.append((left, right))
        continue
      if left < start:
        next_remaining.append((left, start))
      if end < right:
        next_remaining.append((end, right))
    remaining = next_remaining
  return [(start, end) for start, end in remaining if end > start]


def save_agent_timeline(path, events):
  plt = ensure_matplotlib()
  moves = [
      event for event in events
      if event.get("type") == "direct_agent_move"
      and isinstance(event.get("start_time_ns"), int)
      and isinstance(event.get("end_time_ns"), int)
  ]
  if not moves:
    return
  base = min(event["start_time_ns"] for event in moves)
  agents = sorted({event.get("agent", "unknown") for event in moves})
  y_for_agent = {agent: idx for idx, agent in enumerate(agents)}

  fig, ax = plt.subplots(figsize=(12, max(4, 0.8 * len(agents))))
  for event in moves:
    start = (event["start_time_ns"] - base) / 1_000_000_000
    width = (event["end_time_ns"] - event["start_time_ns"]) / 1_000_000_000
    y = y_for_agent[event.get("agent", "unknown")]
    color = "tab:red" if event.get("status") != "ok" else "tab:blue"
    ax.barh(y, width, left=start, height=0.45, color=color, alpha=0.75)
  ax.set_yticks(list(y_for_agent.values()))
  ax.set_yticklabels(agents)
  ax.set_xlabel("Time since run start (s)")
  ax.set_title("Direct Run Agent Move Timeline")
  ax.grid(True, axis="x", alpha=0.25)
  fig.tight_layout()
  fig.savefig(path, dpi=180)
  plt.close(fig)


def save_model_timeline(path, events):
  plt = ensure_matplotlib()
  model_events = [
      event for event in events
      if event.get("type") in ("worker_llm", "worker_embedding")
      and isinstance(event.get("start_time_ns"), int)
      and isinstance(event.get("end_time_ns"), int)
  ]
  if not model_events:
    return
  base = min(event["start_time_ns"] for event in model_events)
  lanes = sorted(
      {
          f"{event.get('agent', 'unknown')}:{event.get('type').replace('worker_', '')}"
          for event in model_events
      }
  )
  y_for_lane = {lane: idx for idx, lane in enumerate(lanes)}
  colors = {"worker_llm": "tab:purple", "worker_embedding": "tab:green"}

  fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(lanes))))
  for event in model_events:
    lane = f"{event.get('agent', 'unknown')}:{event.get('type').replace('worker_', '')}"
    start = (event["start_time_ns"] - base) / 1_000_000_000
    width = (event["end_time_ns"] - event["start_time_ns"]) / 1_000_000_000
    ax.barh(
        y_for_lane[lane],
        width,
        left=start,
        height=0.35,
        color=colors.get(event.get("type"), "tab:gray"),
        alpha=0.75,
    )
  ax.set_yticks(list(y_for_lane.values()))
  ax.set_yticklabels(lanes)
  ax.set_xlabel("Time since first model request (s)")
  ax.set_title("Direct Run LLM / Embedding Timeline")
  ax.grid(True, axis="x", alpha=0.25)
  fig.tight_layout()
  fig.savefig(path, dpi=180)
  plt.close(fig)


def save_agent_phase_timeline(path, events):
  plt = ensure_matplotlib()
  moves = [
      event for event in events
      if event.get("type") == "direct_agent_move"
      and isinstance(event.get("start_time_ns"), int)
      and isinstance(event.get("end_time_ns"), int)
  ]
  model_events = [
      event for event in events
      if event.get("type") in ("worker_llm", "worker_embedding")
      and isinstance(event.get("start_time_ns"), int)
      and isinstance(event.get("end_time_ns"), int)
  ]
  if not moves:
    return

  base = min(event["start_time_ns"] for event in moves)
  agents = sorted({event.get("agent", "unknown") for event in moves})
  y_for_agent = {agent: idx for idx, agent in enumerate(agents)}
  models_by_move = defaultdict(list)
  for event in model_events:
    models_by_move[(event.get("step"), event.get("agent", "unknown"))].append(event)

  colors = {
      "non_model": "tab:green",
      "worker_llm": "tab:blue",
      "worker_embedding": "tab:orange",
  }
  labels_seen = set()
  step_starts = {}
  for move in moves:
    step = move.get("step")
    if step is None:
      continue
    step_starts[step] = min(step_starts.get(step, move["start_time_ns"]), move["start_time_ns"])

  fig, ax = plt.subplots(figsize=(12, max(4, 0.9 * len(agents))))
  for start_ns in sorted(step_starts.values()):
    ax.axvline((start_ns - base) / 1_000_000_000, color="#d0d0d0", linewidth=0.8, alpha=0.6, zorder=0)

  for move in moves:
    step = move.get("step")
    agent = move.get("agent", "unknown")
    y = y_for_agent[agent]
    move_start = move["start_time_ns"]
    move_end = move["end_time_ns"]
    model_parts = []

    for event in models_by_move.get((step, agent), []):
      start = max(move_start, event["start_time_ns"])
      end = min(move_end, event["end_time_ns"])
      if end <= start:
        continue
      model_parts.append((start, end, event.get("type")))

    occupied = [(start, end) for start, end, _ in model_parts]
    for start, end in subtract_intervals(move_start, move_end, occupied):
      label = "Non-model" if "non_model" not in labels_seen else None
      labels_seen.add("non_model")
      ax.barh(
          y,
          (end - start) / 1_000_000_000,
          left=(start - base) / 1_000_000_000,
          height=0.5,
          color=colors["non_model"],
          alpha=0.65,
          label=label,
      )

    for start, end, event_type in sorted(model_parts):
      label_map = {"worker_llm": "LLM", "worker_embedding": "Embedding"}
      label = label_map[event_type] if event_type not in labels_seen else None
      labels_seen.add(event_type)
      ax.barh(
          y,
          (end - start) / 1_000_000_000,
          left=(start - base) / 1_000_000_000,
          height=0.5,
          color=colors[event_type],
          alpha=0.85,
          label=label,
      )

  ax.set_yticks(list(y_for_agent.values()))
  ax.set_yticklabels(agents)
  ax.set_xlabel("Time since run start (s)")
  ax.set_title("Direct Run Agent Timeline by Phase")
  ax.grid(False)
  ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=True)
  fig.tight_layout(rect=[0, 0, 1, 0.93])
  fig.savefig(path, dpi=180)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--perf", required=True)
  parser.add_argument("--out-dir", help="Defaults to <perf dir>/figures.")
  args = parser.parse_args()

  perf_path = os.path.abspath(args.perf)
  events = load_jsonl(perf_path)
  out_dir = args.out_dir or os.path.join(os.path.dirname(perf_path), "figures")
  os.makedirs(out_dir, exist_ok=True)

  save_agent_timeline(os.path.join(out_dir, "direct_agent_move_timeline.png"), events)
  save_model_timeline(os.path.join(out_dir, "direct_model_request_timeline.png"), events)
  save_agent_phase_timeline(os.path.join(out_dir, "direct_agent_phase_timeline.png"), events)
  print(f"figures: {out_dir}")


if __name__ == "__main__":
  main()
