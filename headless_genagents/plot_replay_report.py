"""
Plot figures from a headless replay report and perf JSONL.

Example:
  python plot_replay_report.py \
    --report reports/headless_replay_report_replay_test_headless_trace_1_500_j.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def require_matplotlib():
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in this env with: "
        "pip install matplotlib"
    ) from exc
  return plt


def load_json(path):
  with open(path, "r", encoding="utf-8") as infile:
    return json.load(infile)


def load_jsonl(path):
  events = []
  with open(path, "r", encoding="utf-8") as infile:
    for line in infile:
      if line.strip():
        events.append(json.loads(line))
  return events


def resolve_path(path_value, report_path):
  path = Path(path_value)
  if path.is_absolute():
    return path
  candidates = [
      Path.cwd() / path,
      report_path.parent / path,
      report_path.parent.parent / path,
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate
  return candidates[-1]


def safe_name(value):
  return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def ms_to_min(ms):
  return ms / 1000.0 / 60.0


def shorten(label, width=46):
  if len(label) <= width:
    return label
  return "..." + label[-(width - 3):]


def short_template_name(template):
  if not template:
    return "unknown"
  name = Path(str(template)).name
  if name.endswith(".txt"):
    name = name[:-4]
  for suffix in ("_vMar11", "_v1", "_v2", "_v3"):
    if name.endswith(suffix):
      name = name[:-len(suffix)]
      break
  return name


def savefig(plt, output_dir, name):
  path = output_dir / name
  plt.tight_layout()
  plt.savefig(path, dpi=180, bbox_inches="tight")
  plt.close()
  return path


def latency_values(perf_events, event_type):
  return [
      event.get("latency_ms")
      for event in perf_events
      if event.get("type") == event_type and isinstance(event.get("latency_ms"), (int, float))
  ]


def plot_model_time_breakdown(plt, report, output_dir):
  overall = report.get("overall", {})
  wall = overall.get("wall_time_ms") or 0.0
  llm = overall.get("llm_total_ms") or 0.0
  embedding = overall.get("embedding_total_ms") or 0.0
  non_model = max(0.0, wall - llm - embedding) if wall > 0 else 0.0

  if wall > 0:
    labels = ["LLM", "Embedding", "Non-model"]
    values = [llm, embedding, non_model]
    title = "Wall Time Breakdown"
  else:
    labels = ["LLM", "Embedding"]
    values = [llm, embedding]
    title = "Model Time Breakdown (wall time unavailable)"

  fig, ax = plt.subplots(figsize=(7, 4))
  left = 0
  colors = ["#4C78A8", "#F58518", "#54A24B"]
  for label, value, color in zip(labels, values, colors):
    ax.barh(["time"], [ms_to_min(value)], left=ms_to_min(left), label=label, color=color)
    left += value
  ax.set_xlabel("minutes")
  ax.set_title(title)
  ax.legend(loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, -0.12))
  return savefig(plt, output_dir, "model_time_breakdown.png")


def plot_prompt_template_latency(plt, report, output_dir, top_n):
  data = report.get("llm_by_prompt_template", {})
  items = [
      (template, summary.get("total_ms", 0.0), summary.get("count", 0))
      for template, summary in data.items()
      if summary.get("total_ms", 0.0) > 0
  ]
  items.sort(key=lambda item: item[1], reverse=True)
  items = items[:top_n]
  if not items:
    return None

  labels = [short_template_name(item[0]) for item in reversed(items)]
  values = [ms_to_min(item[1]) for item in reversed(items)]
  counts = [item[2] for item in reversed(items)]

  fig, ax = plt.subplots(figsize=(10, max(4, len(items) * 0.42)))
  bars = ax.barh(labels, values, color="#4C78A8")
  ax.set_xlabel("total LLM latency (minutes)")
  ax.set_title(f"Top {len(items)} Prompt Templates by LLM Time")
  for bar, count in zip(bars, counts):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  n={count}", va="center", fontsize=8)
  return savefig(plt, output_dir, "prompt_template_total_latency.png")


def plot_latency_distribution(plt, perf_events, output_dir):
  llm = latency_values(perf_events, "llm")
  embedding = latency_values(perf_events, "embedding")
  if not llm and not embedding:
    return None

  fig, axes = plt.subplots(1, 2, figsize=(11, 4))
  groups = [("LLM", llm, "#4C78A8"), ("Embedding", embedding, "#F58518")]
  for ax, (title, values, color) in zip(axes, groups):
    values_sec = [value / 1000.0 for value in values]
    if values_sec:
      ax.hist(values_sec, bins=30, color=color, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("latency (seconds)")
    ax.set_ylabel("count")
  fig.suptitle("Model Latency Distribution")
  return savefig(plt, output_dir, "latency_distribution_llm_embedding.png")


def plot_per_step_model_latency(plt, perf_events, output_dir):
  step_move_sum = defaultdict(float)
  step_agent_move_sum = defaultdict(float)
  step_model_sum = defaultdict(float)
  step_agent_model_sum = defaultdict(float)
  step_llm_sum = defaultdict(float)
  step_agent_llm_sum = defaultdict(float)
  step_non_model_sum = defaultdict(float)
  step_agent_non_model_sum = defaultdict(float)
  for event in perf_events:
    step = event.get("step")
    latency = event.get("latency_ms")
    if step is None or not isinstance(latency, (int, float)):
      continue
    step = int(step)
    agent = event.get("agent") or "unknown"
    event_type = event.get("type")
    if event_type == "agent_move_total":
      step_move_sum[step] += latency
      step_agent_move_sum[(step, agent)] += latency
    if event_type in ("llm", "embedding"):
      step_model_sum[step] += latency
      step_agent_model_sum[(step, agent)] += latency
    if event_type == "llm":
      step_llm_sum[step] += latency
      step_agent_llm_sum[(step, agent)] += latency
  for key, move_latency in step_agent_move_sum.items():
    non_model = max(0.0, move_latency - step_agent_model_sum.get(key, 0.0))
    step_agent_non_model_sum[key] = non_model
    step_non_model_sum[key[0]] += non_model
  if not step_move_sum and step_model_sum:
    step_move_sum.update(step_model_sum)
    step_agent_move_sum.update(step_agent_model_sum)
  if not step_move_sum:
    return None

  steps = sorted(step_move_sum)
  max_agent_move = []
  max_agent_model = []
  max_agent_llm = []
  max_agent_non_model = []
  for step in steps:
    max_agent_move.append(
        max(value for (event_step, _agent), value in step_agent_move_sum.items() if event_step == step) / 1000.0
    )
    model_values = [
        value for (event_step, _agent), value in step_agent_model_sum.items()
        if event_step == step
    ]
    max_agent_model.append((max(model_values) if model_values else 0.0) / 1000.0)
    llm_values = [
        value for (event_step, _agent), value in step_agent_llm_sum.items()
        if event_step == step
    ]
    max_agent_llm.append((max(llm_values) if llm_values else 0.0) / 1000.0)
    non_model_values = [
        value for (event_step, _agent), value in step_agent_non_model_sum.items()
        if event_step == step
    ]
    max_agent_non_model.append((max(non_model_values) if non_model_values else 0.0) / 1000.0)

  fig, ax = plt.subplots(figsize=(11, 4))
  ax.plot(steps, [step_move_sum[step] / 1000.0 for step in steps], label="all agents move time", color="#4C78A8", linewidth=1.4)
  ax.plot(steps, max_agent_move, label="slowest agent move time", color="#F58518", linewidth=1.2)
  ax.plot(steps, [step_model_sum.get(step, 0.0) / 1000.0 for step in steps], label="all agents model time", color="#E45756", linewidth=1.0)
  ax.plot(steps, max_agent_model, label="slowest agent model time", color="#72B7B2", linewidth=1.0)
  ax.plot(steps, [step_non_model_sum.get(step, 0.0) / 1000.0 for step in steps], label="all agents non-model time", color="#54A24B", linewidth=1.0)
  ax.plot(steps, max_agent_non_model, label="slowest agent non-model time", color="#B279A2", linewidth=1.0)
  ax.plot(steps, [step_llm_sum.get(step, 0.0) / 1000.0 for step in steps], label="all agents LLM time", color="#FF9DA6", linewidth=0.9, alpha=0.8)
  ax.plot(steps, max_agent_llm, label="slowest agent LLM time", color="#9D755D", linewidth=0.9, alpha=0.8)
  ax.set_xlabel("simulation step")
  ax.set_ylabel("latency (seconds)")
  ax.set_title("Per-Step Move, Model, Non-Model, and LLM Latency")
  ax.legend()
  return savefig(plt, output_dir, "per_step_model_latency.png")


def plot_parallelism_estimate(plt, report, output_dir):
  parallel = report.get("parallelism_estimate", {})
  sequential_move = parallel.get("sequential_agent_move_ms", parallel.get("sequential_model_ms", 0.0))
  agent_move = parallel.get("agent_parallel_agent_move_ms", parallel.get("agent_parallel_model_ms", 0.0))
  sequential_model = parallel.get("sequential_model_ms", 0.0)
  agent_model = parallel.get("agent_parallel_model_ms", 0.0)
  sequential_non_model = parallel.get("sequential_non_model_ms", 0.0)
  agent_non_model = parallel.get("agent_parallel_non_model_ms", 0.0)
  sequential_llm = parallel.get("sequential_llm_ms", 0.0)
  agent_llm = parallel.get("agent_parallel_llm_ms", 0.0)
  if not sequential_move and not agent_move and not sequential_model and not sequential_llm:
    return None

  fig, ax = plt.subplots(figsize=(10, 4))
  labels = ["Move time", "Model time", "LLM time", "Non-model time"]
  sequential_values = [
      ms_to_min(sequential_move),
      ms_to_min(sequential_model),
      ms_to_min(sequential_llm),
      ms_to_min(sequential_non_model),
  ]
  agent_values = [
      ms_to_min(agent_move),
      ms_to_min(agent_model),
      ms_to_min(agent_llm),
      ms_to_min(agent_non_model),
  ]
  x = range(len(labels))
  width = 0.36
  bars1 = ax.bar([i - width / 2 for i in x], sequential_values, width, label="All agents sum", color="#4C78A8")
  bars2 = ax.bar([i + width / 2 for i in x], agent_values, width, label="Slowest agent per step", color="#F58518")
  bars = list(bars1) + list(bars2)
  ax.set_ylabel("minutes")
  move_speedup = parallel.get("estimated_agent_parallel_agent_move_speedup", 0.0)
  model_speedup = parallel.get("estimated_agent_parallel_model_speedup", 0.0)
  llm_speedup = parallel.get("estimated_agent_parallel_llm_speedup", 0.0)
  ax.set_title(f"Agent-Parallel Bounds ({move_speedup:.2f}x move, {model_speedup:.2f}x model, {llm_speedup:.2f}x LLM)")
  ax.set_xticks(list(x))
  ax.set_xticklabels(labels)
  ax.legend()
  for bar in bars:
    value = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom")
  return savefig(plt, output_dir, "parallelism_estimate.png")


def plot_agent_latency(plt, report, output_dir):
  data = report.get("latency_by_agent", {})
  rows = []
  for agent, groups in data.items():
    llm = groups.get("llm", {}).get("total_ms", 0.0)
    embedding = groups.get("embedding", {}).get("total_ms", 0.0)
    if llm or embedding:
      rows.append((agent, llm, embedding))
  rows.sort(key=lambda item: item[1] + item[2], reverse=True)
  if not rows:
    return None

  labels = [row[0] for row in rows]
  llm_values = [ms_to_min(row[1]) for row in rows]
  embedding_values = [ms_to_min(row[2]) for row in rows]

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.bar(labels, llm_values, label="LLM", color="#4C78A8")
  ax.bar(labels, embedding_values, bottom=llm_values, label="Embedding", color="#F58518")
  ax.set_ylabel("minutes")
  ax.set_title("Model Time by Agent")
  ax.legend()
  ax.tick_params(axis="x", rotation=15)
  return savefig(plt, output_dir, "agent_model_latency.png")


def plot_replay_quality(plt, report, output_dir):
  quality = report.get("replay_quality", {})
  keys = [
      "retrieval_canonicalized",
      "reflection_skipped_by_trace",
      "memory_exact",
      "memory_filling_mismatch",
      "memory_core_mismatch",
      "errors",
  ]
  values = [quality.get(key, 0) for key in keys]
  if not any(values):
    return None

  fig, ax = plt.subplots(figsize=(9, 4))
  ax.bar([shorten(key, 24) for key in keys], values, color="#72B7B2")
  ax.set_ylabel("count")
  ax.set_title("Replay Quality and Canonicalization Events")
  ax.tick_params(axis="x", rotation=20)
  return savefig(plt, output_dir, "replay_quality.png")


def plot_trace_coverage(plt, report, output_dir):
  coverage = report.get("trace_coverage", {})
  memory = coverage.get("memory_kinds", {})
  random_fns = coverage.get("random_fns", {})
  if not memory and not random_fns:
    return None

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  axes[0].bar(list(memory.keys()), list(memory.values()), color="#54A24B")
  axes[0].set_title("Memory Add Kinds")
  axes[0].set_ylabel("count")
  axes[1].bar(list(random_fns.keys()), list(random_fns.values()), color="#B279A2")
  axes[1].set_title("Random Function Coverage")
  axes[1].set_ylabel("count")
  fig.suptitle("Trace Coverage")
  return savefig(plt, output_dir, "trace_coverage.png")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", required=True, help="Replay report JSON path.")
  parser.add_argument("--perf", help="Perf JSONL path. Defaults to report['perf_path'].")
  parser.add_argument("--out-dir", help="Output directory. Defaults to <report-dir>/figures.")
  parser.add_argument("--top-n", type=int, default=12)
  args = parser.parse_args()

  plt = require_matplotlib()
  report_path = Path(args.report)
  report = load_json(report_path)
  perf_path = resolve_path(args.perf or report.get("perf_path"), report_path)
  perf_events = load_jsonl(perf_path) if perf_path.exists() else []

  out_dir = Path(args.out_dir) if args.out_dir else report_path.parent / "figures"
  out_dir.mkdir(parents=True, exist_ok=True)

  figures = []
  for maybe_path in [
      plot_model_time_breakdown(plt, report, out_dir),
      plot_prompt_template_latency(plt, report, out_dir, args.top_n),
      plot_latency_distribution(plt, perf_events, out_dir),
      plot_per_step_model_latency(plt, perf_events, out_dir),
      plot_parallelism_estimate(plt, report, out_dir),
      plot_agent_latency(plt, report, out_dir),
      plot_replay_quality(plt, report, out_dir),
      plot_trace_coverage(plt, report, out_dir),
  ]:
    if maybe_path:
      figures.append(maybe_path)

  print(f"Wrote {len(figures)} figures to {out_dir}")
  for path in figures:
    print(path)


if __name__ == "__main__":
  main()
