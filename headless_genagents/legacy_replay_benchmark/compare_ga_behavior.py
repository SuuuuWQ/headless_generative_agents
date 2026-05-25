"""Compare direct-run / replay GA behavior at the (step, agent) level."""
import argparse
import json
import math
import os
from collections import Counter, defaultdict


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
  with open(path, "r", encoding="utf-8") as infile:
    return json.load(infile)


def load_jsonl(path):
  rows = []
  with open(path, "r", encoding="utf-8") as infile:
    for line in infile:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def resolve_path(base_dir, path):
  if not path:
    return None
  if os.path.isabs(path):
    return path
  return os.path.join(SCRIPT_DIR, path)


def output_signature(output):
  if not isinstance(output, dict):
    return None
  return {
      "next_tile": output.get("next_tile"),
      "description": output.get("description"),
  }


def compact_output(output):
  if not isinstance(output, dict):
    return None
  return {
      "next_tile": output.get("next_tile"),
      "pronunciatio": output.get("pronunciatio"),
      "description": output.get("description"),
  }


def llm_api(event):
  api = event.get("api")
  if api:
    return api
  source_event = event.get("source_event")
  if isinstance(source_event, dict) and source_event.get("api"):
    return source_event.get("api")
  return "unknown"


def first_sequence_diff(seq_a, seq_b):
  max_len = max(len(seq_a), len(seq_b))
  for idx in range(max_len):
    item_a = seq_a[idx] if idx < len(seq_a) else None
    item_b = seq_b[idx] if idx < len(seq_b) else None
    if item_a != item_b:
      return idx, item_a, item_b
  return None, None, None


def compact_text_value(value, limit=220):
  if value is None:
    return None
  if isinstance(value, (list, tuple)):
    text = " | ".join(str(item) for item in value)
  else:
    text = str(value)
  text = text.replace("\n", "\\n")
  if len(text) > limit:
    return text[:limit] + "..."
  return text


def event_response_texts(event):
  texts = event.get("response_texts")
  if texts is None:
    source_event = event.get("source_event")
    if isinstance(source_event, dict):
      texts = source_event.get("response_texts")
  if isinstance(texts, list):
    return tuple(texts)
  return None


def event_prompt_record(event):
  record = event.get("prompt_record")
  if record is None:
    source_event = event.get("source_event")
    if isinstance(source_event, dict):
      record = source_event.get("prompt_record")
  return record if isinstance(record, dict) else None


def llm_input_key(api, prompt_record):
  prompt_record = prompt_record or {}
  return (
      api or "unknown",
      prompt_record.get("prompt_template"),
      prompt_record.get("prompt_sha256"),
  )


def llm_call_from_parts(api, prompt_record, response_texts=None):
  return {
      "api": api or "unknown",
      "prompt_template": (prompt_record or {}).get("prompt_template"),
      "prompt_sha256": (prompt_record or {}).get("prompt_sha256"),
      "response_texts": response_texts,
  }


def llm_call_input_key(call):
  return (
      call.get("api") or "unknown",
      call.get("prompt_template"),
      call.get("prompt_sha256"),
  )


def llm_call_io_key(call):
  return (llm_call_input_key(call), call.get("response_texts"))


def compact_llm_input(value):
  if value is None:
    return None
  api, template, prompt_sha = value
  template_name = os.path.basename(template) if template else None
  return {
      "api": api,
      "prompt_template": template_name,
      "prompt_sha256": prompt_sha[:12] if prompt_sha else None,
  }


def compact_llm_io(value):
  if value is None:
    return None
  input_key, response_texts = value
  compact = compact_llm_input(input_key)
  compact["response_texts"] = compact_text_value(response_texts)
  return compact


def llm_diff_kind(reasons):
  reasons = set(reasons or [])
  if "missing_call" in reasons:
    return "missing_call"
  parts = []
  if "api" in reasons:
    parts.append("api")
  if "input" in reasons:
    parts.append("input")
  if "output_text" in reasons:
    parts.append("output")
  if not parts and "input_output" in reasons:
    parts.append("input+output")
  return "+".join(parts) if parts else "unknown"


class Artifact:
  def __init__(self, path):
    self.path = os.path.abspath(path)
    self.config_path = os.path.join(self.path, "config.json")
    self.perf_path = os.path.join(self.path, "perf.jsonl")
    if not os.path.exists(self.config_path):
      raise FileNotFoundError(f"Missing config.json in {self.path}")
    if not os.path.exists(self.perf_path):
      raise FileNotFoundError(f"Missing perf.jsonl in {self.path}")
    self.config = load_json(self.config_path)
    self.mode = self.config.get("mode")
    self.trace_path = self._resolve_trace_path()
    self.trace_events = load_jsonl(self.trace_path) if self.trace_path and os.path.exists(self.trace_path) else []
    self.perf_events = load_jsonl(self.perf_path)
    self.name = os.path.basename(self.path.rstrip(os.sep))
    self.has_actual_llm_texts = False
    self.has_actual_llm_inputs = False
    self.rounds = {}
    self.step_orders = {}
    self.summary = {}
    self._build()

  def _resolve_trace_path(self):
    if self.mode in ("trace_prefix_parallel_replay", "snapshot_prefix_parallel_direct"):
      return resolve_path(self.path, self.config.get("trace"))
    if self.mode == "direct_headless_trace_perf":
      sim = self.config.get("sim")
      if sim:
        return os.path.join(SCRIPT_DIR, "traces", f"trace_{sim}.jsonl")
    return None

  def _build(self):
    rounds = defaultdict(lambda: {
        "output": None,
        "state_signature": None,
        "llm_types": [],
        "llm_texts": [],
        "llm_inputs": [],
        "llm_io": [],
        "embedding_count": 0,
        "retrieval_count": 0,
        "reflection_count": 0,
        "memory_add_count": 0,
        "memory_by_kind": Counter(),
        "move_status": None,
        "move_latency_ms": None,
        "move_start_ns": None,
        "move_end_ns": None,
    })

    if self.mode == "trace_prefix_parallel_replay":
      move_type = "parallel_agent_move"
    elif self.mode == "snapshot_prefix_parallel_direct":
      move_type = "snapshot_parallel_agent_move"
    else:
      move_type = "direct_agent_move"

    for event in self.perf_events:
      event_type = event.get("type")
      step = event.get("step")
      agent = event.get("agent")
      if step is None or agent is None:
        continue
      key = (step, agent)
      row = rounds[key]
      if event_type == move_type:
        output = event.get("output") or event.get("expected_output")
        if output is not None:
          row["output"] = compact_output(output)
          row["state_signature"] = output_signature(output)
        row["move_status"] = event.get("status")
        row["move_latency_ms"] = event.get("latency_ms")
        row["move_start_ns"] = event.get("start_time_ns")
        row["move_end_ns"] = event.get("end_time_ns")
      elif event_type == "worker_llm":
        api = llm_api(event)
        prompt_record = event_prompt_record(event)
        row["llm_types"].append(api)
        row["llm_inputs"].append(llm_input_key(api, prompt_record))
        if prompt_record and prompt_record.get("prompt_sha256"):
          self.has_actual_llm_inputs = True
        response_texts = event_response_texts(event)
        if response_texts is not None:
          row["llm_texts"].append(response_texts)
          self.has_actual_llm_texts = True
        row["llm_io"].append(
            llm_call_io_key(llm_call_from_parts(api, prompt_record, response_texts))
        )
      elif event_type == "worker_embedding":
        row["embedding_count"] += 1
      elif event_type == "worker_retrieval":
        row["retrieval_count"] += 1
      elif event_type == "worker_reflection":
        row["reflection_count"] += 1
      elif event_type == "worker_memory":
        row["memory_add_count"] += 1
        row["memory_by_kind"][event.get("memory_kind")] += 1

    for row in rounds.values():
      if row["reflection_count"] == 0 and row["memory_by_kind"].get("thought", 0) > 0:
        row["reflection_count"] = 1

    has_perf_llm_texts = self.has_actual_llm_texts
    if self.trace_events:
      for event in self.trace_events:
        event_type = event.get("type")
        key = (event.get("step"), event.get("agent"))
        if key[0] is None or key[1] is None:
          continue
        row = rounds[key]
        if event_type == "agent_move_end":
          if row["output"] is None:
            row["output"] = compact_output(event.get("output"))
            row["state_signature"] = output_signature(event.get("output"))
        elif (
            not has_perf_llm_texts
            and
            self.mode == "direct_headless_trace_perf"
            and event_type == "llm_response"
            and event.get("status") == "ok"
        ):
          texts = event.get("canonical_texts")
          if isinstance(texts, list):
            row["llm_texts"].append(tuple(texts))
            self.has_actual_llm_texts = True

    if self.mode == "direct_headless_trace_perf":
      trace_requests = {}
      trace_calls = defaultdict(list)
      for event in self.trace_events:
        event_type = event.get("type")
        if event_type == "llm_request":
          trace_requests[event.get("call_id")] = event
        elif event_type == "llm_response" and event.get("status") == "ok":
          request = trace_requests.get(event.get("call_id"), {})
          step = event.get("step")
          agent = event.get("agent")
          if step is None or agent is None:
            continue
          api = event.get("api") or request.get("api")
          prompt_record = request.get("prompt_record")
          texts = event.get("canonical_texts")
          response_texts = tuple(texts) if isinstance(texts, list) else None
          trace_calls[(step, agent)].append(
              llm_call_from_parts(api, prompt_record, response_texts)
          )

      for key, calls in trace_calls.items():
        row = rounds[key]
        if not row["llm_inputs"] or any(item[1] is None and item[2] is None for item in row["llm_inputs"]):
          row["llm_inputs"] = [llm_call_input_key(call) for call in calls]
          if any(call.get("prompt_sha256") for call in calls):
            self.has_actual_llm_inputs = True
        if not has_perf_llm_texts:
          row["llm_io"] = [llm_call_io_key(call) for call in calls]
        elif not row["llm_io"] or any(item[0][1] is None and item[0][2] is None for item in row["llm_io"]):
          by_text = [call.get("response_texts") for call in calls]
          if by_text == row["llm_texts"]:
            row["llm_io"] = [llm_call_io_key(call) for call in calls]

    self.rounds = dict(rounds)
    self._build_step_orders(move_type)
    self.summary = {
        "llm_requests": sum(1 for e in self.perf_events if e.get("type") == "worker_llm"),
        "llm_response_texts": sum(len(row["llm_texts"]) for row in rounds.values()),
        "llm_input_fingerprints": sum(
            1
            for row in rounds.values()
            for item in row["llm_inputs"]
            if item[2]
        ),
        "embedding_requests": sum(1 for e in self.perf_events if e.get("type") == "worker_embedding"),
        "agent_rounds": len(self.rounds),
    }

  def _build_step_orders(self, move_type):
    by_step = defaultdict(list)
    for event in self.perf_events:
      if event.get("type") != move_type:
        continue
      step = event.get("step")
      agent = event.get("agent")
      end_ns = event.get("end_time_ns")
      if step is None or agent is None or not isinstance(end_ns, int):
        continue
      by_step[step].append((end_ns, agent))

    self.step_orders = {}
    for step, rows in by_step.items():
      rows.sort()
      order = [agent for _, agent in rows]
      rank = {agent: idx for idx, agent in enumerate(order)}
      self.step_orders[step] = {"order": order, "rank": rank}


def compare_artifacts(a, b):
  rounds_a = a.rounds
  rounds_b = b.rounds
  keys_a = set(rounds_a)
  keys_b = set(rounds_b)
  common = sorted(keys_a & keys_b)
  only_a = sorted(keys_a - keys_b)
  only_b = sorted(keys_b - keys_a)

  exact = 0
  same_state = 0
  same_llm_multiset = 0
  same_llm_sequence = 0
  same_llm_text_multiset = 0
  same_llm_text_sequence = 0
  compare_llm_texts = a.has_actual_llm_texts and b.has_actual_llm_texts
  same_llm_input_multiset = 0
  same_llm_input_sequence = 0
  same_llm_io_multiset = 0
  same_llm_io_sequence = 0
  compare_llm_inputs = a.has_actual_llm_inputs and b.has_actual_llm_inputs
  compare_llm_io = compare_llm_inputs and compare_llm_texts
  same_embedding = 0
  same_retrieval = 0
  same_reflection = 0
  same_memory = 0
  mismatch_breakdown = Counter()
  unstable_agents = Counter()
  mismatch_examples = []
  llm_call_stats = {
      "total_a": 0,
      "total_b": 0,
      "paired": 0,
      "extra_a": 0,
      "extra_b": 0,
      "api_same": 0,
      "input_compared": 0,
      "input_same": 0,
      "text_compared": 0,
      "text_same": 0,
      "io_compared": 0,
      "io_same": 0,
      "mismatch_examples": [],
      "mismatches_by_round": defaultdict(list),
  }

  exact_with_rank = 0
  exact_diff_rank = 0
  duration_deltas = []
  compared_rank_rounds = 0
  identical_completion_order_rounds = 0
  all_match_but_order_diff = 0

  common_steps = sorted(set(step for step, _ in common))
  for step in common_steps:
    order_a = a.step_orders.get(step, {})
    order_b = b.step_orders.get(step, {})
    if not order_a or not order_b:
      continue
    common_agents = [agent for agent in order_a["order"] if agent in order_b["rank"]]
    if not common_agents:
      continue
    compared_rank_rounds += 1
    order_a_common = [agent for agent in order_a["order"] if agent in common_agents]
    order_b_common = [agent for agent in order_b["order"] if agent in common_agents]
    if order_a_common == order_b_common:
      identical_completion_order_rounds += 1
    else:
      step_all_match = True
      for agent in common_agents:
        row_a = rounds_a[(step, agent)]
        row_b = rounds_b[(step, agent)]
        if row_a["output"] != row_b["output"]:
          step_all_match = False
          break
      if step_all_match:
        all_match_but_order_diff += 1

  for key in common:
    row_a = rounds_a[key]
    row_b = rounds_b[key]
    step, agent = key
    reasons = []

    count_a = len(row_a["llm_types"])
    count_b = len(row_b["llm_types"])
    paired_llm = min(count_a, count_b)
    llm_call_stats["total_a"] += count_a
    llm_call_stats["total_b"] += count_b
    llm_call_stats["paired"] += paired_llm
    llm_call_stats["extra_a"] += max(0, count_a - count_b)
    llm_call_stats["extra_b"] += max(0, count_b - count_a)
    for idx in range(paired_llm):
      call_reasons = []
      if row_a["llm_types"][idx] == row_b["llm_types"][idx]:
        llm_call_stats["api_same"] += 1
      else:
        call_reasons.append("api")

      input_a = row_a["llm_inputs"][idx] if idx < len(row_a["llm_inputs"]) else None
      input_b = row_b["llm_inputs"][idx] if idx < len(row_b["llm_inputs"]) else None
      text_a = row_a["llm_texts"][idx] if idx < len(row_a["llm_texts"]) else None
      text_b = row_b["llm_texts"][idx] if idx < len(row_b["llm_texts"]) else None
      io_a = row_a["llm_io"][idx] if idx < len(row_a["llm_io"]) else None
      io_b = row_b["llm_io"][idx] if idx < len(row_b["llm_io"]) else None

      if compare_llm_inputs and input_a is not None and input_b is not None:
        llm_call_stats["input_compared"] += 1
        if input_a == input_b:
          llm_call_stats["input_same"] += 1
        else:
          call_reasons.append("input")

      if compare_llm_texts and text_a is not None and text_b is not None:
        llm_call_stats["text_compared"] += 1
        if text_a == text_b:
          llm_call_stats["text_same"] += 1
        else:
          call_reasons.append("output_text")

      if compare_llm_io and io_a is not None and io_b is not None:
        llm_call_stats["io_compared"] += 1
        if io_a == io_b:
          llm_call_stats["io_same"] += 1
        else:
          call_reasons.append("input_output")

      if call_reasons:
        detail = {
            "step": step,
            "agent": agent,
            "llm_index": idx,
            "reasons": call_reasons,
            "api_a": row_a["llm_types"][idx],
            "api_b": row_b["llm_types"][idx],
            "input_a": compact_llm_input(input_a),
            "input_b": compact_llm_input(input_b),
            "text_a": compact_text_value(text_a),
            "text_b": compact_text_value(text_b),
            "io_a": compact_llm_io(io_a),
            "io_b": compact_llm_io(io_b),
        }
        llm_call_stats["mismatches_by_round"][(step, agent)].append(detail)
        if len(llm_call_stats["mismatch_examples"]) < 12:
          llm_call_stats["mismatch_examples"].append(detail)

    if count_a != count_b:
      detail = {
          "step": step,
          "agent": agent,
          "llm_index": paired_llm,
          "reasons": ["missing_call"],
          "api_a": row_a["llm_types"][paired_llm] if count_a > paired_llm else None,
          "api_b": row_b["llm_types"][paired_llm] if count_b > paired_llm else None,
          "input_a": compact_llm_input(row_a["llm_inputs"][paired_llm]) if len(row_a["llm_inputs"]) > paired_llm else None,
          "input_b": compact_llm_input(row_b["llm_inputs"][paired_llm]) if len(row_b["llm_inputs"]) > paired_llm else None,
          "text_a": compact_text_value(row_a["llm_texts"][paired_llm]) if len(row_a["llm_texts"]) > paired_llm else None,
          "text_b": compact_text_value(row_b["llm_texts"][paired_llm]) if len(row_b["llm_texts"]) > paired_llm else None,
          "io_a": compact_llm_io(row_a["llm_io"][paired_llm]) if len(row_a["llm_io"]) > paired_llm else None,
          "io_b": compact_llm_io(row_b["llm_io"][paired_llm]) if len(row_b["llm_io"]) > paired_llm else None,
      }
      llm_call_stats["mismatches_by_round"][(step, agent)].append(detail)
      if len(llm_call_stats["mismatch_examples"]) < 12:
        llm_call_stats["mismatch_examples"].append(detail)

    if row_a["output"] == row_b["output"]:
      exact += 1
    else:
      reasons.append("behavior")

    if row_a["state_signature"] == row_b["state_signature"]:
      same_state += 1
    else:
      reasons.append("state")

    if Counter(row_a["llm_types"]) == Counter(row_b["llm_types"]):
      same_llm_multiset += 1
    else:
      reasons.append("llm_mix")

    if row_a["llm_types"] == row_b["llm_types"]:
      same_llm_sequence += 1
    else:
      if "llm_mix" not in reasons:
        reasons.append("llm_sequence")

    if compare_llm_texts:
      if Counter(row_a["llm_texts"]) == Counter(row_b["llm_texts"]):
        same_llm_text_multiset += 1
      else:
        reasons.append("llm_text_mix")

      if row_a["llm_texts"] == row_b["llm_texts"]:
        same_llm_text_sequence += 1
      else:
        if "llm_text_mix" not in reasons:
          reasons.append("llm_text_sequence")

    if compare_llm_inputs:
      if Counter(row_a["llm_inputs"]) == Counter(row_b["llm_inputs"]):
        same_llm_input_multiset += 1
      else:
        reasons.append("llm_input_mix")

      if row_a["llm_inputs"] == row_b["llm_inputs"]:
        same_llm_input_sequence += 1
      else:
        if "llm_input_mix" not in reasons:
          reasons.append("llm_input_sequence")

    if compare_llm_io:
      if Counter(row_a["llm_io"]) == Counter(row_b["llm_io"]):
        same_llm_io_multiset += 1
      else:
        reasons.append("llm_io_mix")

      if row_a["llm_io"] == row_b["llm_io"]:
        same_llm_io_sequence += 1
      else:
        if "llm_io_mix" not in reasons:
          reasons.append("llm_io_sequence")

    if row_a["embedding_count"] == row_b["embedding_count"]:
      same_embedding += 1
    else:
      reasons.append("embedding")

    if row_a["retrieval_count"] == row_b["retrieval_count"]:
      same_retrieval += 1
    else:
      reasons.append("retrieval")

    if row_a["reflection_count"] == row_b["reflection_count"]:
      same_reflection += 1
    else:
      reasons.append("reflection")

    if row_a["memory_add_count"] == row_b["memory_add_count"]:
      same_memory += 1
    else:
      reasons.append("memory")

    rank_a = a.step_orders.get(step, {}).get("rank", {}).get(agent)
    rank_b = b.step_orders.get(step, {}).get("rank", {}).get(agent)
    if rank_a is not None and rank_b is not None:
      if row_a["output"] == row_b["output"]:
        exact_with_rank += 1
        if rank_a != rank_b:
          exact_diff_rank += 1
      compared_rank_rounds += 0

    dur_a = row_a.get("move_latency_ms")
    dur_b = row_b.get("move_latency_ms")
    if row_a["output"] == row_b["output"] and isinstance(dur_a, (int, float)) and isinstance(dur_b, (int, float)):
      duration_deltas.append(abs(dur_a - dur_b))

    if reasons:
      mismatch_breakdown.update(reasons)
      unstable_agents[agent] += 1
      if len(mismatch_examples) < 12:
        output_a = row_a["output"] or {}
        output_b = row_b["output"] or {}
        if compare_llm_texts:
          llm_text_diff_index, llm_text_a, llm_text_b = first_sequence_diff(
              row_a["llm_texts"], row_b["llm_texts"]
          )
        else:
          llm_text_diff_index, llm_text_a, llm_text_b = None, None, None
        if compare_llm_inputs:
          llm_input_diff_index, llm_input_a, llm_input_b = first_sequence_diff(
              row_a["llm_inputs"], row_b["llm_inputs"]
          )
        else:
          llm_input_diff_index, llm_input_a, llm_input_b = None, None, None
        if compare_llm_io:
          llm_io_diff_index, llm_io_a, llm_io_b = first_sequence_diff(
              row_a["llm_io"], row_b["llm_io"]
          )
        else:
          llm_io_diff_index, llm_io_a, llm_io_b = None, None, None
        mismatch_examples.append(
            {
                "step": step,
                "agent": agent,
                "reasons": reasons,
                "output_a": row_a["output"],
                "output_b": row_b["output"],
                "next_tile_a": output_a.get("next_tile"),
                "next_tile_b": output_b.get("next_tile"),
                "pronunciatio_a": output_a.get("pronunciatio"),
                "pronunciatio_b": output_b.get("pronunciatio"),
                "description_a": output_a.get("description"),
                "description_b": output_b.get("description"),
                "llm_total_a": len(row_a["llm_types"]),
                "llm_total_b": len(row_b["llm_types"]),
                "llm_text_total_a": len(row_a["llm_texts"]),
                "llm_text_total_b": len(row_b["llm_texts"]),
                "llm_text_diff_index": llm_text_diff_index,
                "llm_text_diff_a": compact_text_value(llm_text_a),
                "llm_text_diff_b": compact_text_value(llm_text_b),
                "llm_input_diff_index": llm_input_diff_index,
                "llm_input_diff_a": compact_llm_input(llm_input_a),
                "llm_input_diff_b": compact_llm_input(llm_input_b),
                "llm_io_diff_index": llm_io_diff_index,
                "llm_io_diff_a": compact_llm_io(llm_io_a),
                "llm_io_diff_b": compact_llm_io(llm_io_b),
                "embedding_a": row_a["embedding_count"],
                "embedding_b": row_b["embedding_count"],
                "rank_a": rank_a,
                "rank_b": rank_b,
            }
        )

  return {
      "common": len(common),
      "only_a": len(only_a),
      "only_b": len(only_b),
      "exact": exact,
      "same_state": same_state,
      "same_llm_multiset": same_llm_multiset,
      "same_llm_sequence": same_llm_sequence,
      "same_llm_text_multiset": same_llm_text_multiset,
      "same_llm_text_sequence": same_llm_text_sequence,
      "compare_llm_texts": compare_llm_texts,
      "same_llm_input_multiset": same_llm_input_multiset,
      "same_llm_input_sequence": same_llm_input_sequence,
      "same_llm_io_multiset": same_llm_io_multiset,
      "same_llm_io_sequence": same_llm_io_sequence,
      "compare_llm_inputs": compare_llm_inputs,
      "compare_llm_io": compare_llm_io,
      "same_embedding": same_embedding,
      "same_retrieval": same_retrieval,
      "same_reflection": same_reflection,
      "same_memory": same_memory,
      "llm_call_stats": llm_call_stats,
      "mismatch_breakdown": mismatch_breakdown,
      "unstable_agents": unstable_agents,
      "mismatch_examples": mismatch_examples,
      "compared_rank_rounds": sum(
          1
          for step in common_steps
          if a.step_orders.get(step) and b.step_orders.get(step)
      ),
      "identical_completion_order_rounds": identical_completion_order_rounds,
      "all_match_but_order_diff": all_match_but_order_diff,
      "exact_with_rank": exact_with_rank,
      "exact_diff_rank": exact_diff_rank,
      "mean_abs_duration_delta_ms": (
          sum(duration_deltas) / len(duration_deltas) if duration_deltas else None
      ),
  }


def pct(numerator, denominator):
  if not denominator:
    return "n/a"
  return f"{(100.0 * numerator / denominator):.2f}%"


def print_artifact(label, artifact):
  print(label)
  print(f"- path: {artifact.path}")
  print(f"- mode: {artifact.mode}")
  print(f"- llm requests: {artifact.summary['llm_requests']}")
  text_note = "actual" if artifact.has_actual_llm_texts else "not recorded"
  print(f"- llm response texts: {artifact.summary['llm_response_texts']} ({text_note})")
  input_note = "actual" if artifact.has_actual_llm_inputs else "not recorded"
  print(f"- llm input fingerprints: {artifact.summary['llm_input_fingerprints']} ({input_note})")
  print(f"- embedding requests: {artifact.summary['embedding_requests']}")
  print(f"- agent-round records: {artifact.summary['agent_rounds']}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("artifact_a", help="Run directory with config.json + perf.jsonl")
  parser.add_argument("artifact_b", help="Run directory with config.json + perf.jsonl")
  args = parser.parse_args()

  artifact_a = Artifact(args.artifact_a)
  artifact_b = Artifact(args.artifact_b)
  result = compare_artifacts(artifact_a, artifact_b)

  print("Trace A")
  print_artifact("", artifact_a)
  print("Trace B")
  print_artifact("", artifact_b)
  print()
  print("Behavior comparison")
  print(f"- common agent-rounds: {result['common']}")
  print(f"- only in A: {result['only_a']}")
  print(f"- only in B: {result['only_b']}")
  print(f"- exact behavior match: {result['exact']}/{result['common']} ({pct(result['exact'], result['common'])})")
  print(f"- same output signature: {result['same_state']}/{result['common']} ({pct(result['same_state'], result['common'])})")
  print(f"- same llm type multiset: {result['same_llm_multiset']}/{result['common']} ({pct(result['same_llm_multiset'], result['common'])})")
  print(f"- same llm type sequence: {result['same_llm_sequence']}/{result['common']} ({pct(result['same_llm_sequence'], result['common'])})")
  if result["compare_llm_inputs"]:
    print(f"- same llm input multiset: {result['same_llm_input_multiset']}/{result['common']} ({pct(result['same_llm_input_multiset'], result['common'])})")
    print(f"- same llm input sequence: {result['same_llm_input_sequence']}/{result['common']} ({pct(result['same_llm_input_sequence'], result['common'])})")
  else:
    print("- same llm input multiset: n/a (LLM input fingerprints missing for at least one artifact)")
    print("- same llm input sequence: n/a (LLM input fingerprints missing for at least one artifact)")
  if result["compare_llm_texts"]:
    print(f"- same llm output-text multiset: {result['same_llm_text_multiset']}/{result['common']} ({pct(result['same_llm_text_multiset'], result['common'])})")
    print(f"- same llm output-text sequence: {result['same_llm_text_sequence']}/{result['common']} ({pct(result['same_llm_text_sequence'], result['common'])})")
  else:
    print("- same llm output-text multiset: n/a (actual LLM texts missing for at least one artifact)")
    print("- same llm output-text sequence: n/a (actual LLM texts missing for at least one artifact)")
  if result["compare_llm_io"]:
    print(f"- same llm input+output multiset: {result['same_llm_io_multiset']}/{result['common']} ({pct(result['same_llm_io_multiset'], result['common'])})")
    print(f"- same llm input+output sequence: {result['same_llm_io_sequence']}/{result['common']} ({pct(result['same_llm_io_sequence'], result['common'])})")
  else:
    print("- same llm input+output multiset: n/a (LLM input fingerprints or output texts missing)")
    print("- same llm input+output sequence: n/a (LLM input fingerprints or output texts missing)")
  print(f"- same embedding count: {result['same_embedding']}/{result['common']} ({pct(result['same_embedding'], result['common'])})")
  print(f"- same retrieval count: {result['same_retrieval']}/{result['common']} ({pct(result['same_retrieval'], result['common'])})")
  print(f"- same reflection count: {result['same_reflection']}/{result['common']} ({pct(result['same_reflection'], result['common'])})")
  print(f"- same memory-add count: {result['same_memory']}/{result['common']} ({pct(result['same_memory'], result['common'])})")
  print()
  call_stats = result["llm_call_stats"]
  print("LLM call-level comparison")
  print(f"- llm calls in A: {call_stats['total_a']}")
  print(f"- llm calls in B: {call_stats['total_b']}")
  print(f"- paired by step/agent/order: {call_stats['paired']}")
  print(f"- extra calls only in A/B: {call_stats['extra_a']}/{call_stats['extra_b']}")
  print(
      f"- same api at paired positions: "
      f"{call_stats['api_same']}/{call_stats['paired']} "
      f"({pct(call_stats['api_same'], call_stats['paired'])})"
  )
  if result["compare_llm_inputs"]:
    print(
        f"- same input fingerprint at paired positions: "
        f"{call_stats['input_same']}/{call_stats['input_compared']} "
        f"({pct(call_stats['input_same'], call_stats['input_compared'])})"
    )
  else:
    print("- same input fingerprint at paired positions: n/a (LLM input fingerprints missing)")
  if result["compare_llm_texts"]:
    print(
        f"- same output text at paired positions: "
        f"{call_stats['text_same']}/{call_stats['text_compared']} "
        f"({pct(call_stats['text_same'], call_stats['text_compared'])})"
    )
  else:
    print("- same output text at paired positions: n/a (actual LLM texts missing)")
  if result["compare_llm_io"]:
    print(
        f"- same input+output at paired positions: "
        f"{call_stats['io_same']}/{call_stats['io_compared']} "
        f"({pct(call_stats['io_same'], call_stats['io_compared'])})"
    )
  else:
    print("- same input+output at paired positions: n/a (LLM input fingerprints or output texts missing)")
  print()
  print("LLM differences by step")
  if not call_stats["mismatches_by_round"]:
    print("- none")
  for (step, agent), items in sorted(call_stats["mismatches_by_round"].items()):
    print(f"- step={step} agent={agent}: {len(items)} differing LLM calls")
    for item in items:
      input_a = item["input_a"] or {}
      input_b = item["input_b"] or {}
      template_a = input_a.get("prompt_template")
      template_b = input_b.get("prompt_template")
      sha_a = input_a.get("prompt_sha256")
      sha_b = input_b.get("prompt_sha256")
      if template_a == template_b and sha_a == sha_b:
        input_desc = f"input={template_a}/{sha_a}"
      else:
        input_desc = f"input={template_a}/{sha_a}->{template_b}/{sha_b}"
      print(
          f"  - llm_index={item['llm_index']} "
          f"diff={llm_diff_kind(item['reasons'])} "
          f"api={item['api_a']}->{item['api_b']} "
          f"{input_desc} "
          f"output_a={repr(item['text_a'])} "
          f"output_b={repr(item['text_b'])}"
      )
  print()
  print("Likely scheduling drift")
  print(f"- compared rounds with completion ranks: {result['compared_rank_rounds']}")
  print(
      f"- identical completion order rounds: "
      f"{result['identical_completion_order_rounds']}/{result['compared_rank_rounds']} "
      f"({pct(result['identical_completion_order_rounds'], result['compared_rank_rounds'])})"
  )
  print(
      f"- rounds where all common agents match behavior but completion order differs: "
      f"{result['all_match_but_order_diff']}/{result['compared_rank_rounds']} "
      f"({pct(result['all_match_but_order_diff'], result['compared_rank_rounds'])})"
  )
  print(
      f"- agent-rounds with exact behavior match but different completion rank: "
      f"{result['exact_diff_rank']}/{result['exact_with_rank']} "
      f"({pct(result['exact_diff_rank'], result['exact_with_rank'])})"
  )
  if result["mean_abs_duration_delta_ms"] is not None:
    print(
        "- mean abs step-duration delta among exact-behavior matches: "
        f"{result['mean_abs_duration_delta_ms']:.2f} ms"
    )
  print()
  print("Mismatch breakdown")
  for reason, count in result["mismatch_breakdown"].most_common():
    print(f"- {reason}: {count}")
  print()
  print("Most unstable agents")
  for agent, count in result["unstable_agents"].most_common(10):
    print(f"- {agent}: {count} mismatched rounds")
  print()
  print("Top LLM call mismatches")
  for item in call_stats["mismatch_examples"]:
    print(
        f"- step={item['step']} agent={item['agent']} llm_index={item['llm_index']}: "
        f"reasons={','.join(item['reasons'])} "
        f"api={item['api_a']}->{item['api_b']} "
        f"input_a={repr(item['input_a'])} "
        f"input_b={repr(item['input_b'])} "
        f"text_a={repr(item['text_a'])} "
        f"text_b={repr(item['text_b'])}"
    )
  print()
  print("Top behavior mismatches")
  for item in result["mismatch_examples"]:
    print(
        f"- step={item['step']} agent={item['agent']}: "
        f"reasons={','.join(item['reasons'])} "
        f"next_tile={item['next_tile_a']}->{item['next_tile_b']} "
        f"pronunciatio={repr(item['pronunciatio_a'])}->{repr(item['pronunciatio_b'])} "
        f"description={repr(item['description_a'])}->{repr(item['description_b'])} "
        f"llm_total={item['llm_total_a']}->{item['llm_total_b']} "
        f"first_llm_input_diff={item['llm_input_diff_index']} "
        f"llm_input_a={repr(item['llm_input_diff_a'])} "
        f"llm_input_b={repr(item['llm_input_diff_b'])} "
        f"llm_text_total={item['llm_text_total_a']}->{item['llm_text_total_b']} "
        f"first_llm_text_diff={item['llm_text_diff_index']} "
        f"llm_text_a={repr(item['llm_text_diff_a'])} "
        f"llm_text_b={repr(item['llm_text_diff_b'])} "
        f"first_llm_io_diff={item['llm_io_diff_index']} "
        f"llm_io_a={repr(item['llm_io_diff_a'])} "
        f"llm_io_b={repr(item['llm_io_diff_b'])} "
        f"embedding={item['embedding_a']}->{item['embedding_b']} "
        f"rank={item['rank_a']}->{item['rank_b']}"
    )


if __name__ == "__main__":
  main()
