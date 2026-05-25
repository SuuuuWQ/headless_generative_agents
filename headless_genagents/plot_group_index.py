"""
Visualize conflict-grouping results for a headless GA run.

Examples:
  python plot_group_index.py \
    ../environment/frontend_server/storage/direct_trace_perf_n25_10_a/group_index.jsonl \
    --step 0

  python plot_group_index.py \
    ../environment/frontend_server/storage/direct_trace_perf_n25_10_a/group_index.jsonl \
    --step 0 \
    --focus-agent "Jane Moreno"
"""
import argparse
import json
import os
import sys
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(SCRIPT_DIR, "runs")


def require_matplotlib():
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
  except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install it with: pip install matplotlib"
    ) from exc
  return plt, Rectangle


def resolve_path(path, base_dir=None):
  if not path:
    return None
  if os.path.exists(path):
    return path
  if base_dir:
    candidate = os.path.join(base_dir, path)
    if os.path.exists(candidate):
      return candidate
  normalized = path.replace("\\", "/")
  if normalized.startswith("/mnt/") and len(normalized) > 6:
    drive = normalized[5]
    rest = normalized[7:].replace("/", os.sep)
    candidate = f"{drive.upper()}:{os.sep}{rest}"
    if os.path.exists(candidate):
      return candidate
  return path


def infer_group_index_path(group_index_or_sim):
  resolved = resolve_path(group_index_or_sim)
  looks_like_path = (
      os.path.exists(resolved)
      or group_index_or_sim.endswith(".jsonl")
      or os.path.sep in group_index_or_sim
      or "/" in group_index_or_sim
      or "\\" in group_index_or_sim
  )
  if looks_like_path:
    return resolved

  sim = group_index_or_sim
  candidates = [
      os.path.join(RUN_ROOT, sim, "group_index.jsonl"),
      os.path.join(
          SCRIPT_DIR,
          "..",
          "environment",
          "frontend_server",
          "storage",
          sim,
          "group_index.jsonl",
      ),
  ]
  for candidate in candidates:
    if os.path.exists(candidate):
      return os.path.abspath(candidate)
  raise SystemExit(
      f"Could not infer group_index for sim '{sim}':\n  "
      + "\n  ".join(candidates)
  )


def read_jsonl(path):
  records = []
  with open(path, "r", encoding="utf-8") as infile:
    for line in infile:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return records


def load_group_index(path):
  meta = {}
  steps = {}
  for record in read_jsonl(path):
    if record.get("type") == "group_index_meta":
      meta = record
    elif record.get("type") == "group_step":
      steps[int(record["step"])] = record
  return meta, steps


def load_position_index(path):
  meta = {}
  steps = {}
  for record in read_jsonl(path):
    if record.get("type") == "position_index_meta":
      meta = record
    elif record.get("type") == "position_step":
      steps[int(record["step"])] = record
  return meta, steps


def short_name(name):
  parts = name.split()
  if len(parts) >= 2:
    return f"{parts[0][0]}. {parts[-1]}"
  return name


def object_label(subject):
  if not subject:
    return ""
  return str(subject).split(":")[-1]


def color_for_groups(plt, groups):
  palette = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors)
  color_by_agent = {}
  for idx, group in enumerate(groups):
    color = palette[idx % len(palette)]
    for agent in group:
      color_by_agent[agent] = color
  return color_by_agent


def group_id_by_agent(groups):
  mapping = {}
  for idx, group in enumerate(groups):
    for agent in group:
      mapping[agent] = idx + 1
  return mapping


def load_maze(maze_name):
  if not maze_name:
    return None
  try:
    from maze import Maze
  except ImportError:
    return None
  return Maze(maze_name)


def arena_tiles(maze, arena_path):
  if not maze or not arena_path:
    return set()
  if arena_path in maze.address_tiles:
    return set(maze.address_tiles[arena_path])
  tiles = set()
  for y in range(maze.maze_height):
    for x in range(maze.maze_width):
      if maze.get_tile_path((x, y), "arena") == arena_path:
        tiles.add((x, y))
  return tiles


def row_spans(tiles):
  rows = defaultdict(list)
  for x, y in tiles:
    rows[y].append(x)
  spans = []
  for y, xs in rows.items():
    xs = sorted(xs)
    start = prev = xs[0]
    for x in xs[1:]:
      if x == prev + 1:
        prev = x
      else:
        spans.append((start, prev, y))
        start = prev = x
    spans.append((start, prev, y))
  return spans


def short_arena_path(path):
  parts = [part for part in str(path).split(":") if part]
  if len(parts) >= 2:
    return ":".join(parts[-2:])
  return parts[-1] if parts else "(empty arena)"


def is_real_arena_path(path):
  parts = str(path or "").split(":")
  return len(parts) == 3 and bool(parts[1]) and bool(parts[2])


def percepts_for_plot(step_record, focus_agent=None):
  percepts = step_record.get("percepts") or {}
  if focus_agent:
    return {focus_agent: percepts.get(focus_agent, [])}
  return percepts


def draw_step(
    group_record,
    position_record,
    agent_params,
    output_path,
    focus_agent=None,
    show_labels=True,
    show_vision=True,
    show_arena=True,
    all_arenas=False,
    label_arenas=False,
    show_percepts=True,
    show_edges=True,
    invert_y=True,
    crop_padding=12,
    maze=None,
):
  plt, Rectangle = require_matplotlib()

  agents = position_record.get("agents") or {}
  groups = group_record.get("groups") or []
  color_by_agent = color_for_groups(plt, groups)
  group_ids = group_id_by_agent(groups)

  fig, ax = plt.subplots(figsize=(13, 8))
  ax.set_title(
      f"Interaction groups, step {group_record.get('step')}"
      + (f" | focus: {focus_agent}" if focus_agent else "")
  )
  ax.set_aspect("equal", adjustable="box")
  ax.grid(True, linewidth=0.35, alpha=0.25)

  xs = []
  ys = []

  if show_arena and maze is not None:
    arena_agents = defaultdict(list)
    for agent, info in agents.items():
      arena_path = info.get("arena_path")
      if is_real_arena_path(arena_path):
        arena_agents[arena_path].append(agent)

    highlighted_arenas = set(arena_agents)
    for arena_path in maze.address_tiles:
      if arena_path.count(":") == 2:
        arena_agents.setdefault(arena_path, [])

    for arena_path, members in arena_agents.items():
      tiles = arena_tiles(maze, arena_path)
      if not tiles:
        continue
      is_highlighted = arena_path in highlighted_arenas
      color = color_by_agent.get(members[0], "#d8d8d8") if members else "#d8d8d8"
      alpha = 0.50 if is_highlighted else 0.16
      linewidth = 0.5 if is_highlighted else 0.2
      for x0, x1, y in row_spans(tiles):
        rect = Rectangle(
            (x0 - 0.5, y - 0.5),
            x1 - x0 + 1,
            1,
            facecolor=color,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=0,
        )
        ax.add_patch(rect)
      tile_xs = [tile[0] for tile in tiles]
      tile_ys = [tile[1] for tile in tiles]
      xs.extend(tile_xs)
      ys.extend(tile_ys)
      if label_arenas and is_highlighted:
        cx = sum(tile_xs) / len(tile_xs)
        cy = sum(tile_ys) / len(tile_ys)
        ax.text(
            cx,
            cy,
            short_arena_path(arena_path),
            fontsize=6,
            color="#444444",
            ha="center",
            va="center",
            alpha=0.55,
            zorder=0.5,
        )

  if show_vision:
    for agent, info in agents.items():
      if focus_agent and agent != focus_agent:
        continue
      x = info["x"]
      y = info["y"]
      vision_r = (agent_params.get(agent) or {}).get("vision_r", 0)
      color = color_by_agent.get(agent, "black")
      rect = Rectangle(
          (x - vision_r - 0.5, y - vision_r - 0.5),
          2 * vision_r + 1,
          2 * vision_r + 1,
          fill=False,
          edgecolor=color,
          linewidth=1.2,
          linestyle="--",
          alpha=0.45,
      )
      ax.add_patch(rect)
      xs.extend([x - vision_r, x + vision_r])
      ys.extend([y - vision_r, y + vision_r])

  if show_percepts and group_record.get("percepts"):
    for observer, items in percepts_for_plot(group_record, focus_agent).items():
      observer_pos = agents.get(observer)
      if not observer_pos:
        continue
      observer_color = color_by_agent.get(observer, "black")
      for item in items:
        tile = item.get("tile")
        if not tile or len(tile) != 2:
          continue
        x, y = tile
        xs.append(x)
        ys.append(y)
        if item.get("kind") == "agent":
          target = item.get("subject")
          if target == observer:
            continue
          ax.scatter(
              [x],
              [y],
              s=58,
              facecolors="none",
              edgecolors=observer_color,
              linewidths=1.4,
              alpha=0.95,
              zorder=3,
          )
          ax.plot(
              [observer_pos["x"], x],
              [observer_pos["y"], y],
              color=observer_color,
              linewidth=1.0,
              alpha=0.45,
              zorder=1,
          )
        else:
          ax.scatter(
              [x],
              [y],
              s=28,
              marker="x",
              color=observer_color if focus_agent else "#555555",
              alpha=0.55 if focus_agent else 0.25,
              zorder=2,
          )
          if focus_agent:
            ax.text(
                x + 0.25,
                y + 0.25,
                object_label(item.get("subject")),
                fontsize=6,
                color="#555555",
                alpha=0.85,
            )

  if show_edges:
    for edge in group_record.get("edges") or []:
      a = edge.get("a")
      b = edge.get("b")
      if focus_agent and focus_agent not in (a, b):
        continue
      if a not in agents or b not in agents:
        continue
      ax.plot(
          [agents[a]["x"], agents[b]["x"]],
          [agents[a]["y"], agents[b]["y"]],
          color="#222222",
          linewidth=2.0,
          alpha=0.5,
          zorder=1,
      )

  for agent, info in agents.items():
    x = info["x"]
    y = info["y"]
    xs.append(x)
    ys.append(y)
    color = color_by_agent.get(agent, "#999999")
    alpha = 1.0 if not focus_agent or agent == focus_agent else 0.35
    ax.scatter([x], [y], s=44, color=color, edgecolors="black", linewidths=0.55, alpha=alpha, zorder=4)
    if show_labels:
      label = f"G{group_ids.get(agent, '?')} {short_name(agent)}"
      ax.text(x + 0.35, y - 0.35, label, fontsize=7, color="#111111", alpha=alpha, zorder=5)

  if xs and ys:
    ax.set_xlim(min(xs) - crop_padding, max(xs) + crop_padding)
    ax.set_ylim(min(ys) - crop_padding, max(ys) + crop_padding)
  if invert_y:
    ax.invert_yaxis()
  ax.set_xlabel("tile x")
  ax.set_ylabel("tile y")

  stats = group_record.get("stats") or {}
  if stats:
    caption = (
        f"agents={stats.get('agent_count')}  "
        f"groups={stats.get('group_count')}  "
        f"edges={stats.get('edge_count')}"
    )
    ax.text(
        0.01,
        0.01,
        caption,
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 3},
    )

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  fig.savefig(output_path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("group_index", help="Sim name or path to group_index.jsonl.")
  parser.add_argument("--position-index", help="Path to position_index.jsonl. Defaults to meta path or sibling file.")
  parser.add_argument("--step", type=int, help="Step to plot. Defaults to first available step.")
  parser.add_argument("--all", action="store_true", help="Plot every step.")
  parser.add_argument("--focus-agent", help="Only draw one agent's vision/percepts, while keeping all agents visible.")
  parser.add_argument("--output-dir", help="Defaults to <group_index_dir>/figures/group_checks.")
  parser.add_argument("--no-labels", action="store_true")
  parser.add_argument("--no-arena", action="store_true")
  parser.add_argument("--label-arenas", action="store_true")
  parser.add_argument(
      "--all-arenas",
      action="store_true",
      help="Deprecated: all arenas are drawn by default.",
  )
  parser.add_argument("--no-vision", action="store_true")
  parser.add_argument("--no-percepts", action="store_true")
  parser.add_argument("--no-edges", action="store_true")
  parser.add_argument("--no-invert-y", action="store_true")
  parser.add_argument("--crop-padding", type=float, default=12)
  args = parser.parse_args()

  group_index = infer_group_index_path(args.group_index)
  group_dir = os.path.dirname(os.path.abspath(group_index))
  group_meta, group_steps = load_group_index(group_index)
  if not group_steps:
    raise SystemExit(f"No group_step records found in {group_index}")

  position_path = args.position_index or group_meta.get("position_index")
  if not position_path:
    position_path = os.path.join(group_dir, "position_index.jsonl")
  position_path = resolve_path(position_path, group_dir)
  _, position_steps = load_position_index(position_path)
  maze = load_maze(group_meta.get("maze"))

  if args.all:
    steps = sorted(group_steps)
  elif args.step is not None:
    steps = [args.step]
  else:
    steps = [min(group_steps)]

  output_dir = args.output_dir or os.path.join(group_dir, "figures", "group_checks")
  written = []
  for step in steps:
    if step not in group_steps:
      print(f"[skip] step {step}: no group record", file=sys.stderr)
      continue
    if step not in position_steps:
      print(f"[skip] step {step}: no position record", file=sys.stderr)
      continue
    suffix = f"_focus_{args.focus_agent.replace(' ', '_')}" if args.focus_agent else ""
    output_path = os.path.join(output_dir, f"group_step_{step:04d}{suffix}.png")
    draw_step(
        group_steps[step],
        position_steps[step],
        group_meta.get("agent_params") or {},
        output_path,
        focus_agent=args.focus_agent,
        show_labels=not args.no_labels,
        show_arena=not args.no_arena,
        all_arenas=args.all_arenas,
        label_arenas=args.label_arenas,
        show_vision=not args.no_vision,
        show_percepts=not args.no_percepts,
        show_edges=not args.no_edges,
        invert_y=not args.no_invert_y,
        crop_padding=args.crop_padding,
        maze=maze,
    )
    written.append(output_path)

  for path in written:
    print(path)


if __name__ == "__main__":
  main()
