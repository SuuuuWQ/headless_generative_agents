"""Plot timeline figures for grouped trace-driven LLM benchmark logs."""
import argparse
import os
from collections import defaultdict

from plot_direct_perf import ensure_matplotlib, load_jsonl


def _timed(events, event_type):
  return [
      event
      for event in events
      if event.get("type") == event_type
      and isinstance(event.get("start_time_ns"), int)
      and isinstance(event.get("end_time_ns"), int)
  ]


def _base_time(events):
  timed = [
      event
      for event in events
      if isinstance(event.get("start_time_ns"), int)
  ]
  return min((event["start_time_ns"] for event in timed), default=0)


def _sec(ns, base):
  return (ns - base) / 1_000_000_000


def _group_color(plt, group_id):
  palette = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors)
  try:
    index = int(group_id)
  except (TypeError, ValueError):
    index = 0
  return palette[index % len(palette)]


def save_group_timeline(path, events):
  plt = ensure_matplotlib()
  groups = _timed(events, "grouped_group_round")
  if not groups:
    return

  base = _base_time(groups)
  lanes = sorted(
      {
          f"step {event.get('step')} / group {event.get('group_id')}"
          for event in groups
      },
      key=lambda item: tuple(int(part) for part in item.replace("step ", "").replace("group ", "").split(" / ")),
  )
  y_for_lane = {lane: idx for idx, lane in enumerate(lanes)}

  fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(lanes))))
  for event in groups:
    lane = f"step {event.get('step')} / group {event.get('group_id')}"
    start = _sec(event["start_time_ns"], base)
    width = (event["end_time_ns"] - event["start_time_ns"]) / 1_000_000_000
    color = _group_color(plt, event.get("group_id"))
    edgecolor = "tab:red" if event.get("error_count") else color
    ax.barh(
        y_for_lane[lane],
        width,
        left=start,
        height=0.35,
        color=color,
        edgecolor=edgecolor,
        linewidth=1.2 if event.get("error_count") else 0.4,
        alpha=0.78,
    )
    label = f"{event.get('llm_calls', 0)} calls"
    ax.text(start + width, y_for_lane[lane], " " + label, va="center", fontsize=6, color="#333333")

  ax.set_yticks(list(y_for_lane.values()))
  ax.set_yticklabels(lanes, fontsize=7)
  ax.set_xlabel("Time since first group start (s)")
  ax.set_title("Grouped Trace LLM Benchmark: Group Timeline")
  ax.grid(False)
  fig.tight_layout()
  fig.savefig(path, dpi=180)
  plt.close(fig)


def save_agent_llm_timeline(path, events):
  plt = ensure_matplotlib()
  agent_rounds = _timed(events, "grouped_agent_round")
  steps = _timed(events, "grouped_step_round")
  if not agent_rounds:
    return

  base = _base_time(agent_rounds)
  agents = sorted({event.get("agent", "unknown") for event in agent_rounds})
  y_for_agent = {agent: idx for idx, agent in enumerate(agents)}

  fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(agents))))
  for step in steps:
    ax.axvline(
        _sec(step["start_time_ns"], base),
        color="#d0d0d0",
        linewidth=0.8,
        alpha=0.7,
        zorder=0,
    )
  if steps:
    last_end = max(step["end_time_ns"] for step in steps)
    ax.axvline(
        _sec(last_end, base),
        color="#d0d0d0",
        linewidth=0.8,
        alpha=0.7,
        zorder=0,
    )

  for event in agent_rounds:
    agent = event.get("agent", "unknown")
    start = _sec(event["start_time_ns"], base)
    width = (event["end_time_ns"] - event["start_time_ns"]) / 1_000_000_000
    color = _group_color(plt, event.get("group_id"))
    has_error = bool(event.get("error_count"))
    ax.barh(
        y_for_agent[agent],
        width,
        left=start,
        height=0.42,
        color=color,
        edgecolor="tab:red" if has_error else "none",
        linewidth=0.7 if has_error else 0,
        alpha=0.68,
        zorder=2,
    )

  ax.set_yticks(list(y_for_agent.values()))
  ax.set_yticklabels(agents, fontsize=7)
  ax.set_xlabel("Time since first LLM call (s)")
  ax.set_title("Grouped Trace LLM Benchmark: Agent LLM Timeline")
  ax.grid(False)
  fig.tight_layout(rect=[0, 0, 1, 0.94])
  fig.savefig(path, dpi=180)
  plt.close(fig)


def save_step_summary(path, events):
  plt = ensure_matplotlib()
  steps = sorted(_timed(events, "grouped_step_round"), key=lambda item: item.get("step", 0))
  if not steps:
    return

  x = [event.get("step") for event in steps]
  wall = [event.get("latency_ms", 0.0) / 1000.0 for event in steps]
  max_group = [event.get("max_group_ms", 0.0) / 1000.0 for event in steps]
  sum_group = [event.get("sum_group_ms", 0.0) / 1000.0 for event in steps]
  calls = [event.get("llm_calls", 0) for event in steps]

  fig, ax1 = plt.subplots(figsize=(11, 4.5))
  ax1.plot(x, wall, marker="o", linewidth=1.5, label="step wall time", color="tab:blue")
  ax1.plot(x, max_group, marker="o", linewidth=1.1, label="slowest group", color="tab:orange")
  ax1.plot(x, sum_group, marker="o", linewidth=1.0, label="sum group time", color="tab:green", alpha=0.8)
  ax1.set_xlabel("Simulation step")
  ax1.set_ylabel("Seconds")
  ax1.grid(True, axis="y", alpha=0.25)

  ax2 = ax1.twinx()
  ax2.bar(x, calls, alpha=0.18, color="tab:gray", label="LLM calls")
  ax2.set_ylabel("LLM calls")

  lines1, labels1 = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
  ax1.set_title("Grouped Trace LLM Benchmark: Step Summary")
  fig.tight_layout()
  fig.savefig(path, dpi=180)
  plt.close(fig)


def write_figures(perf_path, out_dir=None):
  perf_path = os.path.abspath(perf_path)
  events = load_jsonl(perf_path)
  out_dir = out_dir or os.path.join(os.path.dirname(perf_path), "figures")
  os.makedirs(out_dir, exist_ok=True)
  save_group_timeline(os.path.join(out_dir, "grouped_group_timeline.png"), events)
  save_agent_llm_timeline(os.path.join(out_dir, "grouped_agent_llm_timeline.png"), events)
  step_summary_path = os.path.join(out_dir, "grouped_step_summary.png")
  if os.path.exists(step_summary_path):
    os.remove(step_summary_path)
  return out_dir


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--perf", required=True)
  parser.add_argument("--out-dir", help="Defaults to <perf dir>/figures.")
  args = parser.parse_args()

  out_dir = write_figures(args.perf, args.out_dir)
  print(f"figures: {out_dir}")


if __name__ == "__main__":
  main()
