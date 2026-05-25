"""
Run the copied headless GA under trace-guided replay.

This is the first replay version that keeps the real headless GA control flow.
It executes expensive/nondeterministic nodes, records timing, then returns the
canonical trace result to the GA wherever the trace has the semantic result.

Usage:
  python legacy_replay_benchmark/headless_replay_runner.py \
    --trace traces/trace_headless_trace_test.jsonl
"""
import argparse
import builtins
import datetime
import hashlib
import json
import os
import random
import sys
import threading
import time
import traceback

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
  sys.path.insert(0, PARENT_DIR)

import openai

import sglang_openai_patch as sglang_patch


class ReplayMismatch(Exception):
  pass


class ReplayController:
  def __init__(self, trace_path, strict=True, perf_path=None):
    self.trace_path = trace_path
    self.strict = strict
    self.events = []
    self.by_event_id = {}
    self.seq_by_type = {}
    self.next_index_by_type = {}
    self.context = threading.local()
    self.event_counters = {}
    self.prompt_by_text = {}
    self.sim_base_step = None
    self.sim_base_time = None
    self.sec_per_step = None
    self.is_running = False
    self.perf_path = perf_path
    self.perf_file = None
    self.perf_seq = 0
    self.perf_counts = {}
    self.perf_status_counts = {}
    self.perf_latency_by_type = {}
    self.perf_events = []
    self.warnings = []
    self.load_trace()
    if perf_path:
      os.makedirs(os.path.dirname(perf_path) or ".", exist_ok=True)
      self.perf_file = open(perf_path, "w", encoding="utf-8", buffering=1)

  def load_trace(self):
    with open(self.trace_path, "r", encoding="utf-8") as infile:
      for line in infile:
        if not line.strip():
          continue
        event = json.loads(line)
        self.events.append(event)
        event_id = event.get("event_id")
        if event_id:
          self.by_event_id.setdefault(event_id, []).append(event)
        self.seq_by_type.setdefault(event.get("type"), []).append(event)

  def close(self):
    if self.perf_file:
      self.perf_file.close()

  def warn(self, message):
    self.warnings.append(message)
    print(f"[replay warning] {message}")

  def perf(self, **event):
    if not self.perf_file:
      return
    now_ns = time.time_ns()
    if "end_time_ns" not in event:
      event["end_time_ns"] = now_ns
    if "start_time_ns" not in event:
      latency_ms = event.get("latency_ms")
      if isinstance(latency_ms, (int, float)):
        event["start_time_ns"] = int(event["end_time_ns"] - (latency_ms * 1_000_000))
    self.perf_seq += 1
    event = {"seq": self.perf_seq, **self.safe(event)}
    event_type = event.get("type")
    if event_type:
      self.perf_counts[event_type] = self.perf_counts.get(event_type, 0) + 1
      latency_ms = event.get("latency_ms")
      if isinstance(latency_ms, (int, float)):
        self.perf_latency_by_type[event_type] = (
            self.perf_latency_by_type.get(event_type, 0.0) + latency_ms
        )
    status = event.get("status")
    if status:
      self.perf_status_counts[status] = self.perf_status_counts.get(status, 0) + 1
    self.perf_events.append(event)
    self.perf_file.write(json.dumps(event, ensure_ascii=True) + "\n")

  def safe(self, value):
    if isinstance(value, (str, int, float, bool)) or value is None:
      return value
    if isinstance(value, datetime.datetime):
      return value.isoformat(sep=" ")
    if isinstance(value, datetime.date):
      return value.isoformat()
    if isinstance(value, tuple):
      return [self.safe(item) for item in value]
    if isinstance(value, list):
      return [self.safe(item) for item in value]
    if isinstance(value, set):
      return sorted([self.safe(item) for item in value], key=str)
    if isinstance(value, dict):
      return {str(key): self.safe(val) for key, val in value.items()}
    if hasattr(value, "node_id"):
      return self.safe(self.node(value))
    return repr(value)

  def node(self, node):
    return {
        "node_id": getattr(node, "node_id", None),
        "node_type": getattr(node, "type", None),
        "created": getattr(node, "created", None),
        "expiration": getattr(node, "expiration", None),
        "subject": getattr(node, "subject", None),
        "predicate": getattr(node, "predicate", None),
        "object": getattr(node, "object", None),
        "description": getattr(node, "description", None),
        "embedding_key": getattr(node, "embedding_key", None),
        "poignancy": getattr(node, "poignancy", None),
        "keywords": getattr(node, "keywords", None),
        "filling": getattr(node, "filling", None),
    }

  def current_agent(self):
    return getattr(self.context, "agent", None)

  def current_step(self):
    return getattr(self.context, "step", None)

  def set_agent_context(self, name, step):
    self.context.agent = name
    self.context.step = step
    self.context.local_event_ids = {}

  def clear_agent_context(self):
    self.context.agent = None
    self.context.step = None
    self.context.local_event_ids = {}
    self.context.last_llm_text = None

  def set_last_llm_text(self, text):
    self.context.last_llm_text = text
    self.context.recent_llm_text = text

  def pop_last_llm_text(self):
    text = getattr(self.context, "last_llm_text", None)
    self.context.last_llm_text = None
    return text

  def recent_llm_text(self):
    return getattr(self.context, "recent_llm_text", None)

  def set_sim_clock(self, step, curr_time, sec_per_step):
    self.sim_base_step = step
    self.sim_base_time = curr_time
    self.sec_per_step = sec_per_step

  def estimate_step(self, curr_time):
    if self.sim_base_step is None or self.sim_base_time is None:
      return None
    if not isinstance(curr_time, datetime.datetime):
      return None
    elapsed = int((curr_time - self.sim_base_time).total_seconds())
    return self.sim_base_step + int(elapsed / self.sec_per_step)

  def event_id(self, kind, step=None, agent=None, label=None):
    if step is None:
      step = self.current_step()
    if agent is None:
      agent = self.current_agent()
    agent_part = str(agent).replace(" ", "_") if agent else "global"
    key = (step, agent_part, kind)
    self.event_counters[key] = self.event_counters.get(key, 0) + 1
    parts = [str(step), agent_part, kind, str(self.event_counters[key])]
    if label:
      parts.append(str(label).replace("/", "_").replace("\\", "_"))
    return "|".join(parts)

  def global_event_id(self, kind, step=None, label=None):
    return self.event_id(kind, step=step, agent="global", label=label)

  def record_prompt(self, prompt, curr_input, prompt_lib_file):
    prompt = prompt or ""
    template_path = str(prompt_lib_file)
    prompt_id = f"prompt_{len(self.prompt_by_text) + 1}"
    record = {
        "prompt_id": prompt_id,
        "prompt_sha256": self.sha256_text(prompt),
        "prompt_template": template_path,
        "prompt_template_path": template_path,
        "prompt_template_sha256": self.sha256_file(template_path),
        "inputs": self.safe(curr_input),
    }
    self.prompt_by_text[prompt] = record

  def prompt_record(self, prompt):
    record = self.prompt_by_text.get(prompt)
    if record:
      return record
    if isinstance(prompt, str) and prompt.startswith('"""\n'):
      marker = '\n"""\nOutput the response to the prompt above in json.'
      end = prompt.find(marker)
      if end >= 0:
        return self.prompt_by_text.get(prompt[4:end])
    return None

  def compact_prompt_record(self, prompt):
    record = self.prompt_record(prompt)
    if not record:
      return None
    return {
        "prompt_id": record["prompt_id"],
        "prompt_sha256": record["prompt_sha256"],
        "prompt_template": record["prompt_template"],
        "prompt_template_sha256": record["prompt_template_sha256"],
    }

  def sha256_file(self, path):
    try:
      with open(path, "rb") as infile:
        return hashlib.sha256(infile.read()).hexdigest()
    except Exception:
      return None

  def sha256_text(self, text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

  def consume(self, event_type, event_id):
    queue = self.by_event_id.get(event_id, [])
    for idx, event in enumerate(queue):
      if event.get("type") == event_type and not event.get("_consumed"):
        event["_consumed"] = True
        return event
    message = f"Missing trace event type={event_type} event_id={event_id}"
    if self.strict:
      raise ReplayMismatch(message)
    self.warn(message)
    return None

  def consume_next(self, event_type, **criteria):
    events = self.seq_by_type.get(event_type, [])
    start = self.next_index_by_type.get(event_type, 0)
    for idx in range(start, len(events)):
      event = events[idx]
      if event.get("_consumed"):
        continue
      matched = True
      for key, expected in criteria.items():
        if expected is None:
          continue
        if event.get(key) != expected:
          matched = False
          break
      if matched:
        event["_consumed"] = True
        self.next_index_by_type[event_type] = idx + 1
        return event
    message = f"Missing next trace event type={event_type} criteria={criteria}"
    if self.strict:
      raise ReplayMismatch(message)
      self.warn(message)
    return None

  def has_unconsumed(self, event_type, **criteria):
    events = self.seq_by_type.get(event_type, [])
    for event in events:
      if event.get("_consumed"):
        continue
      matched = True
      for key, expected in criteria.items():
        if expected is None:
          continue
        if event.get(key) != expected:
          matched = False
          break
      if matched:
        return True
    return False

  def check_equal(self, label, actual, expected):
    actual_safe = self.safe(actual)
    expected_safe = self.safe(expected)
    if actual_safe != expected_safe:
      message = (
          f"{label} mismatch\n"
          f"actual={json.dumps(actual_safe, ensure_ascii=True)[:1000]}\n"
          f"expected={json.dumps(expected_safe, ensure_ascii=True)[:1000]}"
      )
      if self.strict:
        raise ReplayMismatch(message)
      self.warn(message)

  def simulation_init_event(self):
    events = self.seq_by_type.get("simulation_init", [])
    return events[0] if events else None


REPLAY = None


def load_trace_for_defaults(trace_path):
  events = []
  with open(trace_path, "r", encoding="utf-8") as infile:
    for line in infile:
      if not line.strip():
        continue
      events.append(json.loads(line))
  return events


def first_default_event(events, event_type):
  for event in events:
    if event.get("type") == event_type:
      return event
  return None


def _to_compat(value):
  return sglang_patch._to_openai_compat(value)


def _node_id(value):
  if isinstance(value, dict):
    return value.get("node_id")
  return getattr(value, "node_id", None)


def _rel_ctx_signature(value):
  if not isinstance(value, dict):
    return None
  if "curr_event" not in value:
    return None
  return (
      _node_id(value.get("curr_event")),
      tuple(_node_id(node) for node in value.get("events", [])),
      tuple(_node_id(node) for node in value.get("thoughts", [])),
  )


def _hydrate_random_result(recorded, candidates):
  """Map JSON-safe random trace values back to live runtime objects."""
  if isinstance(candidates, range):
    candidates = list(candidates)
  try:
    candidate_list = list(candidates)
  except TypeError:
    candidate_list = None

  if candidate_list is not None:
    recorded_sig = _rel_ctx_signature(recorded)
    if recorded_sig is not None:
      for candidate in candidate_list:
        if _rel_ctx_signature(candidate) == recorded_sig:
          return candidate

    recorded_node_id = _node_id(recorded)
    if recorded_node_id:
      for candidate in candidate_list:
        if _node_id(candidate) == recorded_node_id:
          return candidate

    for candidate in candidate_list:
      if REPLAY.safe(candidate) == recorded:
        return candidate

  return recorded


def _parse_datetime(value):
  if value is None or isinstance(value, datetime.datetime):
    return value
  if isinstance(value, str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
      try:
        return datetime.datetime.strptime(value, fmt)
      except ValueError:
        pass
  return value


def _rebuild_memory_indexes(memory):
  memory.id_to_node = {}
  memory.kw_to_event = {}
  memory.kw_to_thought = {}
  memory.kw_to_chat = {}
  memory.kw_strength_event = {}
  memory.kw_strength_thought = {}

  def add_kw(target, node):
    for keyword in [kw.lower() for kw in node.keywords]:
      target.setdefault(keyword, []).append(node)

  def add_strength(target, node):
    if f"{node.predicate} {node.object}" == "is idle":
      return
    for keyword in [kw.lower() for kw in node.keywords]:
      target[keyword] = target.get(keyword, 0) + 1

  for node in memory.seq_event:
    if not hasattr(node, "node_id"):
      continue
    if not isinstance(node.node_id, str):
      node.node_id = json.dumps(REPLAY.safe(node.node_id), ensure_ascii=True)
    memory.id_to_node[node.node_id] = node
    add_kw(memory.kw_to_event, node)
    add_strength(memory.kw_strength_event, node)
  for node in memory.seq_thought:
    if not hasattr(node, "node_id"):
      continue
    if not isinstance(node.node_id, str):
      node.node_id = json.dumps(REPLAY.safe(node.node_id), ensure_ascii=True)
    memory.id_to_node[node.node_id] = node
    add_kw(memory.kw_to_thought, node)
    add_strength(memory.kw_strength_thought, node)
  for node in memory.seq_chat:
    if not hasattr(node, "node_id"):
      continue
    if not isinstance(node.node_id, str):
      node.node_id = json.dumps(REPLAY.safe(node.node_id), ensure_ascii=True)
    memory.id_to_node[node.node_id] = node
    add_kw(memory.kw_to_chat, node)


def _canonicalize_memory_node(memory, node, record):
  if not record:
    return node

  old_embedding_key = getattr(node, "embedding_key", None)
  old_embedding = memory.embeddings.get(old_embedding_key)
  old_node_id = getattr(node, "node_id", None)

  node.node_id = record.get("node_id", node.node_id)
  node.type = record.get("node_type", node.type)
  node.created = _parse_datetime(record.get("created", node.created))
  node.expiration = _parse_datetime(record.get("expiration", node.expiration))
  node.subject = record.get("subject", node.subject)
  node.predicate = record.get("predicate", node.predicate)
  node.object = record.get("object", node.object)
  node.description = record.get("description", node.description)
  node.embedding_key = record.get("embedding_key", node.embedding_key)
  node.poignancy = record.get("poignancy", node.poignancy)
  node.keywords = set(record.get("keywords", node.keywords))
  node.filling = record.get("filling", node.filling)

  if node.embedding_key not in memory.embeddings and old_embedding is not None:
    memory.embeddings[node.embedding_key] = old_embedding
  if old_node_id and old_node_id != node.node_id:
    memory.id_to_node.pop(old_node_id, None)
  _rebuild_memory_indexes(memory)
  return node


def _make_memory_node(memory, record):
  from persona.memory_structures.associative_memory import ConceptNode

  node_type = record.get("node_type") or record.get("type") or "event"
  if node_type == "event":
    type_count = len(memory.seq_event) + 1
    seq = memory.seq_event
  elif node_type == "thought":
    type_count = len(memory.seq_thought) + 1
    seq = memory.seq_thought
  else:
    type_count = len(memory.seq_chat) + 1
    seq = memory.seq_chat

  node_count = len(memory.id_to_node) + 1
  node = ConceptNode(
      record.get("node_id", f"node_{node_count}"),
      node_count,
      type_count,
      node_type,
      0,
      _parse_datetime(record.get("created")),
      _parse_datetime(record.get("expiration")),
      record.get("subject"),
      record.get("predicate"),
      record.get("object"),
      record.get("description"),
      record.get("embedding_key"),
      record.get("poignancy"),
      set(record.get("keywords", [])),
      record.get("filling"),
  )
  seq[0:0] = [node]
  if node.embedding_key and node.embedding_key not in memory.embeddings:
    memory.embeddings[node.embedding_key] = [1e-8] * 1536
  _rebuild_memory_indexes(memory)
  return node


def _node_from_record(memory, record):
  if not record:
    return None
  node_id = record.get("node_id")
  node = memory.id_to_node.get(node_id)
  if node is None:
    node = _make_memory_node(memory, record)
  else:
    _canonicalize_memory_node(memory, node, record)
  return node


def _compact_memory_for_save(memory):
  nodes = []
  seen = set()
  for seq in (memory.seq_chat, memory.seq_thought, memory.seq_event):
    for node in reversed(seq):
      if not hasattr(node, "node_id") or not hasattr(node, "type"):
        continue
      if id(node) in seen:
        continue
      seen.add(id(node))
      nodes.append(node)

  old_to_new = {}
  for index, node in enumerate(nodes, start=1):
    if not isinstance(node.node_id, str):
      node.node_id = json.dumps(REPLAY.safe(node.node_id), ensure_ascii=True)
    old_to_new[node.node_id] = f"node_{index}"

  type_counts = {"event": 0, "thought": 0, "chat": 0}
  for index, node in enumerate(nodes, start=1):
    node.node_count = index
    node.node_id = old_to_new[node.node_id]
    type_counts[node.type] = type_counts.get(node.type, 0) + 1
    node.type_count = type_counts[node.type]
    if isinstance(node.filling, list):
      new_filling = []
      for item in node.filling:
        if isinstance(item, str):
          new_filling.append(old_to_new.get(item, item))
        elif hasattr(item, "node_id"):
          item_id = item.node_id
          if not isinstance(item_id, str):
            item_id = json.dumps(REPLAY.safe(item_id), ensure_ascii=True)
          new_filling.append(old_to_new.get(item_id, item_id))
        else:
          new_filling.append(json.dumps(REPLAY.safe(item), ensure_ascii=True))
      node.filling = new_filling

  memory.seq_event = [node for node in reversed(nodes) if node.type == "event"]
  memory.seq_thought = [node for node in reversed(nodes) if node.type == "thought"]
  memory.seq_chat = [node for node in reversed(nodes) if node.type == "chat"]
  _rebuild_memory_indexes(memory)


def _compact_server_memories(server):
  for persona in getattr(server, "personas", {}).values():
    _compact_memory_for_save(persona.a_mem)


def _completion_response_from_texts(real_response, texts):
  response = dict(real_response) if isinstance(real_response, dict) else {}
  response["choices"] = [{"text": text} for text in texts]
  return _to_compat(response)


def _chat_response_from_texts(real_response, texts):
  response = dict(real_response) if isinstance(real_response, dict) else {}
  response["choices"] = [
      {"message": {"role": "assistant", "content": text}} for text in texts
  ]
  return _to_compat(response)


def _response_usage(response):
  try:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
  except Exception:
    usage = {}
  return {
      "prompt_tokens": usage.get("prompt_tokens"),
      "completion_tokens": usage.get("completion_tokens"),
      "total_tokens": usage.get("total_tokens"),
  }


def _warn_or_check(label, actual, expected):
  try:
    REPLAY.check_equal(label, actual, expected)
  except ReplayMismatch as exc:
    # GPT_request/ChatGPT_request in the original GA swallows broad
    # exceptions and turns them into TOKEN LIMIT/ChatGPT ERROR strings.
    # A replay mismatch here should stay visible as a warning while the
    # canonical trace response still drives the GA state.
    REPLAY.warn(str(exc))


def _prompt_record_for_compare(record):
  if not record:
    return record
  return {
      "prompt_sha256": record.get("prompt_sha256"),
      "prompt_template": record.get("prompt_template"),
      "prompt_template_sha256": record.get("prompt_template_sha256"),
  }


def _wrap_gpt_structure_module(module):
  def wrap_text_request(func_name):
    original = getattr(module, func_name, None)
    if not original or getattr(original, "_replay_wrapped", False):
      return

    def replay_text_request(*args, **kwargs):
      REPLAY.pop_last_llm_text()
      try:
        result = original(*args, **kwargs)
      except Exception:
        canonical = REPLAY.pop_last_llm_text()
        if canonical is not None:
          return canonical
        raise
      canonical = REPLAY.pop_last_llm_text()
      if canonical is not None:
        return canonical
      return result

    replay_text_request._replay_wrapped = True
    setattr(module, func_name, replay_text_request)

  for func_name in (
      "GPT_request",
      "ChatGPT_request",
      "GPT4_request",
      "ChatGPT_single_request",
  ):
    wrap_text_request(func_name)


def _json_output_text(text):
  if text is None:
    return None
  try:
    start = text.index("{")
    end = text.rindex("}") + 1
    parsed = json.loads(text[start:end])
    return parsed.get("output")
  except Exception:
    return None


def _clean_canonical_text(text, prompt, func_validate, func_clean_up):
  if text is None or not func_validate or not func_clean_up:
    return None
  try:
    if func_validate(text, prompt=prompt):
      return func_clean_up(text, prompt=prompt)
  except Exception:
    return None
  return None


def _wrap_safe_generate_module(module):
  def wrap_safe(func_name, parser):
    original = getattr(module, func_name, None)
    if not original or getattr(original, "_replay_wrapped", False):
      return

    def replay_safe_generate(*args, **kwargs):
      REPLAY.context.recent_llm_text = None
      error = None
      try:
        result = original(*args, **kwargs)
      except Exception as exc:
        result = None
        error = exc
      canonical = REPLAY.recent_llm_text()
      if canonical is None:
        if error:
          raise error
        return result
      cleaned = parser(canonical, args, kwargs)
      if cleaned is not None:
        return cleaned
      if error:
        raise error
      return result

    replay_safe_generate._replay_wrapped = True
    setattr(module, func_name, replay_safe_generate)

  def parse_completion(canonical, args, kwargs):
    prompt = args[0] if args else kwargs.get("prompt")
    func_validate = args[4] if len(args) > 4 else kwargs.get("func_validate")
    func_clean_up = args[5] if len(args) > 5 else kwargs.get("func_clean_up")
    return _clean_canonical_text(canonical, prompt, func_validate, func_clean_up)

  def parse_chat(canonical, args, kwargs):
    prompt = args[0] if args else kwargs.get("prompt")
    example_output = args[1] if len(args) > 1 else kwargs.get("example_output")
    special_instruction = args[2] if len(args) > 2 else kwargs.get("special_instruction")
    func_validate = args[5] if len(args) > 5 else kwargs.get("func_validate")
    func_clean_up = args[6] if len(args) > 6 else kwargs.get("func_clean_up")
    decorated_prompt = '"""\n' + prompt + '\n"""\n'
    decorated_prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    decorated_prompt += "Example output json:\n"
    decorated_prompt += '{"output": "' + str(example_output) + '"}'
    output_text = _json_output_text(canonical)
    return _clean_canonical_text(output_text, decorated_prompt, func_validate, func_clean_up)

  def parse_old_chat(canonical, args, kwargs):
    prompt = args[0] if args else kwargs.get("prompt")
    func_validate = args[3] if len(args) > 3 else kwargs.get("func_validate")
    func_clean_up = args[4] if len(args) > 4 else kwargs.get("func_clean_up")
    return _clean_canonical_text(canonical, prompt, func_validate, func_clean_up)

  wrap_safe("safe_generate_response", parse_completion)
  wrap_safe("ChatGPT_safe_generate_response", parse_chat)
  wrap_safe("ChatGPT_safe_generate_response_OLD", parse_old_chat)


def _wrap_generate_prompt_module(module):
  original_generate_prompt = getattr(module, "generate_prompt", None)
  if not original_generate_prompt or getattr(
      original_generate_prompt, "_replay_wrapped", False
  ):
    return

  def replay_generate_prompt(curr_input, prompt_lib_file):
    prompt = original_generate_prompt(curr_input, prompt_lib_file)
    REPLAY.record_prompt(prompt, curr_input, prompt_lib_file)
    return prompt

  replay_generate_prompt._replay_wrapped = True
  module.generate_prompt = replay_generate_prompt


def _wrap_prompt_result_module(module):
  for func_name in dir(module):
    if not func_name.startswith("run_gpt_prompt_"):
      continue
    original = getattr(module, func_name, None)
    if not callable(original) or getattr(original, "_replay_wrapped", False):
      continue

    def make_replayed(name, original_func):
      def replay_prompt_function(*args, **kwargs):
        error = None
        try:
          real_result = original_func(*args, **kwargs)
        except Exception as exc:
          real_result = None
          error = exc
        try:
          event = REPLAY.consume_next(
              "prompt_result",
              agent=REPLAY.current_agent(),
              step=REPLAY.current_step(),
              prompt_function=name,
          )
        except Exception as exc:
          if real_result is not None:
            return real_result
          if not isinstance(exc, ReplayMismatch):
            raise
          raise
        if event:
          expected = event.get("result")
          if real_result is not None:
            try:
              REPLAY.check_equal(f"prompt_result {name}", real_result, expected)
              status = "exact"
            except ReplayMismatch as exc:
              REPLAY.warn(str(exc))
              status = "canonicalized"
          else:
            status = "canonicalized"
          REPLAY.perf(
              type="prompt_result",
              event_id=event.get("event_id"),
              agent=REPLAY.current_agent(),
              step=REPLAY.current_step(),
              prompt_function=name,
              status=status,
              error=repr(error) if error else None,
          )
          return expected
        if error:
          raise error
        return real_result

      replay_prompt_function._replay_wrapped = True
      return replay_prompt_function

    setattr(module, func_name, make_replayed(func_name, original))


def _wrap_loaded_replay_modules():
  gpt_structure = sys.modules.get("persona.prompt_template.gpt_structure")
  if not gpt_structure:
    return

  _wrap_generate_prompt_module(gpt_structure)
  _wrap_gpt_structure_module(gpt_structure)
  _wrap_safe_generate_module(gpt_structure)

  replay_names = (
      "generate_prompt",
      "GPT_request",
      "ChatGPT_request",
      "GPT4_request",
      "ChatGPT_single_request",
      "safe_generate_response",
      "ChatGPT_safe_generate_response",
      "ChatGPT_safe_generate_response_OLD",
  )
  for module in list(sys.modules.values()):
    module_name = getattr(module, "__name__", "")
    if not module_name.startswith("persona.prompt_template."):
      continue
    if module is gpt_structure:
      continue
    for name in replay_names:
      if hasattr(module, name) and hasattr(gpt_structure, name):
        setattr(module, name, getattr(gpt_structure, name))
    if module_name == "persona.prompt_template.run_gpt_prompt":
      _wrap_prompt_result_module(module)


def _install_import_replay_hooks():
  original_import = builtins.__import__

  def replay_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)
    if name == "persona.prompt_template.gpt_structure":
      _wrap_generate_prompt_module(module)
      _wrap_gpt_structure_module(module)
    if name == "persona.prompt_template.run_gpt_prompt":
      _wrap_prompt_result_module(module)
    return module

  builtins.__import__ = replay_import
  _wrap_loaded_replay_modules()


def _install_openai_replay_hooks():
  original_completion_create = openai.Completion.create
  original_chat_create = openai.ChatCompletion.create
  original_embedding_create = openai.Embedding.create

  def replay_completion_create(*args, **kwargs):
    prompt = kwargs.get("prompt")
    if prompt is None and args:
      prompt = args[0]
    prompt_record = REPLAY.compact_prompt_record(prompt)
    request_event = REPLAY.consume_next(
        "llm_request",
        api="completion",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
    )
    event_id = (
        request_event.get("event_id")
        if request_event else
        REPLAY.event_id("llm", label="completion")
    )
    if request_event and prompt_record:
      expected = request_event.get("prompt_record") or (request_event.get("request") or {}).get("prompt_record")
      if expected:
        _warn_or_check(
            "prompt_record",
            _prompt_record_for_compare(prompt_record),
            _prompt_record_for_compare(expected),
        )

    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    error = None
    try:
      real_response = original_completion_create(*args, **kwargs)
    except Exception as exc:
      real_response = {"choices": []}
      status = "error"
      error = repr(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000
    end_time_ns = time.time_ns()
    response_event = REPLAY.consume("llm_response", event_id)
    texts = (response_event or {}).get("canonical_texts", [])
    if texts:
      REPLAY.set_last_llm_text(texts[0])
    trace_prompt_record = None
    if request_event:
      trace_prompt_record = (
          request_event.get("prompt_record")
          or (request_event.get("request") or {}).get("prompt_record")
      )
    REPLAY.perf(
        type="llm",
        event_id=event_id,
        api="completion",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
        prompt_record=trace_prompt_record or prompt_record,
        actual_prompt_record=prompt_record,
        response_texts=texts,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=elapsed_ms,
        status=status,
        error=error,
        usage=_response_usage(real_response),
    )
    if response_event is None or not texts:
      return _to_compat(real_response)
    return _completion_response_from_texts(real_response, texts)

  def replay_chat_create(*args, **kwargs):
    messages = kwargs.get("messages") or []
    prompt = None
    prompt_record = None
    label = "chat"
    if messages and isinstance(messages, list):
      prompt = str(messages[0].get("content", ""))
      prompt_record = REPLAY.compact_prompt_record(prompt)
      label = hashlib.sha256(
          prompt.encode("utf-8")
      ).hexdigest()[:12]
    if prompt_record:
      label = prompt_record.get("prompt_template") or label
    request_event = REPLAY.consume_next(
        "llm_request",
        api="chat",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
    )
    event_id = (
        request_event.get("event_id")
        if request_event else
        REPLAY.event_id("llm", label=label)
    )
    if request_event and prompt_record:
      expected = request_event.get("prompt_record") or (request_event.get("request") or {}).get("prompt_record")
      if expected:
        _warn_or_check(
            "prompt_record",
            _prompt_record_for_compare(prompt_record),
            _prompt_record_for_compare(expected),
        )

    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    error = None
    try:
      real_response = original_chat_create(*args, **kwargs)
    except Exception as exc:
      real_response = {"choices": []}
      status = "error"
      error = repr(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000
    end_time_ns = time.time_ns()
    response_event = REPLAY.consume("llm_response", event_id)
    texts = (response_event or {}).get("canonical_texts", [])
    if texts:
      REPLAY.set_last_llm_text(texts[0])
    trace_prompt_record = None
    if request_event:
      trace_prompt_record = (
          request_event.get("prompt_record")
          or (request_event.get("request") or {}).get("prompt_record")
      )
    REPLAY.perf(
        type="llm",
        event_id=event_id,
        api="chat",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
        prompt_record=trace_prompt_record or prompt_record,
        actual_prompt_record=prompt_record,
        response_texts=texts,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=elapsed_ms,
        status=status,
        error=error,
        usage=_response_usage(real_response),
    )
    if response_event is None or not texts:
      return _to_compat(real_response)
    return _chat_response_from_texts(real_response, texts)

  def replay_embedding_create(*args, **kwargs):
    request_event = REPLAY.consume_next(
        "embedding_request",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
    )
    event_id = (
        request_event.get("event_id")
        if request_event else
        REPLAY.event_id("embedding")
    )

    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    error = None
    try:
      real_response = original_embedding_create(*args, **kwargs)
    except Exception as exc:
      real_response = {"data": []}
      status = "error"
      error = repr(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000
    end_time_ns = time.time_ns()
    response_event = REPLAY.consume("embedding_response", event_id)
    if response_event:
      summary = response_event.get("canonical_summary", {})
      real_data = real_response.get("data", []) if isinstance(real_response, dict) else []
      real_dims = []
      for item in real_data:
        embedding = item.get("embedding")
        real_dims.append(len(embedding) if embedding is not None else None)
      REPLAY.check_equal("embedding count", len(real_data), summary.get("count"))
      REPLAY.check_equal("embedding dimensions", real_dims, summary.get("dimensions"))
    REPLAY.perf(
        type="embedding",
        event_id=event_id,
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=elapsed_ms,
        status=status,
        error=error,
    )
    # By design, replay traces do not store embedding vectors. The real
    # embedding is returned here, while retrieval is canonicalized by node_id.
    return _to_compat(real_response)

  openai.Completion.create = staticmethod(replay_completion_create)
  openai.ChatCompletion.create = staticmethod(replay_chat_create)
  openai.Embedding.create = staticmethod(replay_embedding_create)


def _install_random_replay_hooks():
  original_choice = random.choice
  original_choices = random.choices
  original_randint = random.randint
  original_sample = random.sample

  def replay_choice(seq):
    real_result = original_choice(seq)
    event = REPLAY.consume_next(
        "random_result",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
        fn="choice",
    )
    if not event:
      return real_result
    return _hydrate_random_result(event.get("result"), seq)

  def replay_choices(population, weights=None, cum_weights=None, k=1):
    real_result = original_choices(
        population, weights=weights, cum_weights=cum_weights, k=k
    )
    event = REPLAY.consume_next(
        "random_result",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
        fn="choices",
    )
    if not event:
      return real_result
    recorded = event.get("result")
    if isinstance(recorded, list):
      return [_hydrate_random_result(item, population) for item in recorded]
    return recorded

  def replay_randint(a, b):
    real_result = original_randint(a, b)
    event = REPLAY.consume_next(
        "random_result",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
        fn="randint",
    )
    return (event or {}).get("result", real_result)

  def replay_sample(population, k, counts=None):
    if counts is None:
      real_result = original_sample(population, k)
    else:
      real_result = original_sample(population, k, counts=counts)
    event = REPLAY.consume_next(
        "random_result",
        agent=REPLAY.current_agent(),
        step=REPLAY.current_step(),
        fn="sample",
    )
    if not event:
      return real_result
    recorded = event.get("result")
    if isinstance(recorded, list):
      return [_hydrate_random_result(item, population) for item in recorded]
    return recorded

  random.choice = replay_choice
  random.choices = replay_choices
  random.randint = replay_randint
  random.sample = replay_sample


def _install_class_replay_hooks():
  original_build_class = builtins.__build_class__

  def replay_build_class(func, name, *args, **kwargs):
    cls = original_build_class(func, name, *args, **kwargs)

    if name == "ReverieServer":
      original_init = cls.__init__
      original_start_server = cls.start_server

      def replay_init(self, fork_sim_code, sim_code):
        original_init(self, fork_sim_code, sim_code)
        REPLAY.set_sim_clock(
            getattr(self, "step", None),
            getattr(self, "curr_time", None),
            getattr(self, "sec_per_step", None),
        )
        event_id = REPLAY.global_event_id(
            "simulation",
            step=getattr(self, "step", None),
            label="init",
        )
        event = REPLAY.consume("simulation_init", event_id)
        if event:
          REPLAY.check_equal("fork_sim_code", fork_sim_code, event.get("fork_sim_code"))
          REPLAY.check_equal("start_step", getattr(self, "step", None), event.get("step"))

      def replay_start_server(self, int_counter):
        start_id = REPLAY.global_event_id(
            "run",
            step=getattr(self, "step", None),
            label="start",
        )
        event = REPLAY.consume("run_start", start_id)
        if event:
          REPLAY.check_equal("requested_steps", int_counter, event.get("requested_steps"))
        REPLAY.is_running = True
        failed = False
        try:
          return original_start_server(self, int_counter)
        except Exception:
          failed = True
          raise
        finally:
          REPLAY.is_running = False
          if not failed:
            end_id = REPLAY.global_event_id(
                "run",
                step=getattr(self, "step", None),
                label="end",
            )
            event = REPLAY.consume("run_end", end_id)
            if event:
              REPLAY.check_equal("end_step", getattr(self, "step", None), event.get("end_step"))

      cls.__init__ = replay_init
      cls.start_server = replay_start_server

    elif name == "Persona":
      original_move = cls.move
      original_retrieve = cls.retrieve
      original_reflect = cls.reflect

      def replay_move(self, maze, personas, curr_tile, curr_time):
        curr_step = REPLAY.estimate_step(curr_time)
        agent_name = getattr(self, "name", None)
        move_started_at = time.perf_counter()
        move_status = "ok"
        REPLAY.set_agent_context(getattr(self, "name", None), curr_step)
        move_event_id = REPLAY.event_id("agent_move")
        start = REPLAY.consume("agent_move_start", move_event_id)
        if start:
          REPLAY.check_equal("agent_move curr_tile", curr_tile, start.get("curr_tile"))
        try:
          output = original_move(self, maze, personas, curr_tile, curr_time)
          end = REPLAY.consume("agent_move_end", move_event_id)
          if end:
            expected_output = end.get("output")
            actual_output = {
                "next_tile": output[0],
                "pronunciatio": output[1],
                "description": output[2],
            }
            REPLAY.check_equal("agent_move output", actual_output, expected_output)
          return output
        except Exception:
          move_status = "error"
          raise
        finally:
          REPLAY.perf(
              type="agent_move_total",
              status=move_status,
              agent=agent_name,
              step=curr_step,
              event_id=move_event_id,
              latency_ms=(time.perf_counter() - move_started_at) * 1000.0,
          )
          REPLAY.clear_agent_context()

      def replay_retrieve(self, perceived):
        output = original_retrieve(self, perceived)
        event_id = REPLAY.event_id("retrieval")
        event = REPLAY.consume("retrieval_result", event_id)
        if event:
          canonical = {}
          focal_points = event.get("focal_points", {})
          for focal_point, groups in focal_points.items():
            actual_groups = output.get(focal_point, {})
            actual_curr_event = actual_groups.get("curr_event")
            curr_event = actual_curr_event
            if groups.get("curr_event"):
              curr_event = _node_from_record(self.a_mem, groups.get("curr_event"))

            events = []
            event_records = (groups.get("_event_records") or {})
            for item in groups.get("events", []):
              node_id = item.get("node_id") if isinstance(item, dict) else item
              record = item if isinstance(item, dict) else event_records.get(node_id)
              node = self.a_mem.id_to_node.get(node_id)
              if node is None and record:
                node = _node_from_record(self.a_mem, record)
              if node is None:
                node = _make_memory_node(
                    self.a_mem,
                    {
                        "node_id": node_id,
                        "node_type": "event",
                        "created": getattr(self.scratch, "curr_time", None),
                        "subject": node_id,
                        "predicate": "be",
                        "object": "unknown",
                        "description": node_id,
                        "embedding_key": node_id,
                        "poignancy": 1,
                        "keywords": [node_id],
                        "filling": [],
                    },
                )
              events.append(node)

            thoughts = []
            thought_records = (groups.get("_thought_records") or {})
            for item in groups.get("thoughts", []):
              node_id = item.get("node_id") if isinstance(item, dict) else item
              record = item if isinstance(item, dict) else thought_records.get(node_id)
              node = self.a_mem.id_to_node.get(node_id)
              if node is None and record:
                node = _node_from_record(self.a_mem, record)
              if node is None:
                node = _make_memory_node(
                    self.a_mem,
                    {
                        "node_id": node_id,
                        "node_type": "thought",
                        "created": getattr(self.scratch, "curr_time", None),
                        "subject": node_id,
                        "predicate": "be",
                        "object": "unknown",
                        "description": node_id,
                        "embedding_key": node_id,
                        "poignancy": 1,
                        "keywords": [node_id],
                        "filling": [],
                    },
                )
              thoughts.append(node)

            canonical[focal_point] = {
                "curr_event": curr_event,
                "events": events,
                "thoughts": thoughts,
            }

          REPLAY.perf(
              type="retrieval",
              event_id=event_id,
              agent=REPLAY.current_agent(),
              step=REPLAY.current_step(),
              status="canonicalized",
          )
          return canonical
        return output

      cls.move = replay_move
      cls.retrieve = replay_retrieve

      def replay_reflect(self):
        if not REPLAY.is_running:
          return original_reflect(self)
        has_trace_thought = REPLAY.has_unconsumed(
            "memory_add",
            agent=getattr(self, "name", None),
            step=REPLAY.current_step(),
            memory_kind="thought",
        )
        if not has_trace_thought:
          REPLAY.perf(
              type="reflection",
              agent=getattr(self, "name", None),
              step=REPLAY.current_step(),
              status="skipped_by_trace",
          )
          return None
        return original_reflect(self)

      cls.reflect = replay_reflect

    elif name == "AssociativeMemory":
      for method_name in ("add_event", "add_thought", "add_chat"):
        if not hasattr(cls, method_name):
          continue
        original_method = getattr(cls, method_name)

        def make_replay(method_name, original_method):
          def replay_add(self, *args, **kwargs):
            node = original_method(self, *args, **kwargs)
            if not REPLAY.is_running:
              return node
            event_id = REPLAY.event_id(
                "memory",
                label=method_name.replace("add_", ""),
            )
            event = REPLAY.consume("memory_add", event_id)
            if event:
              actual_node = REPLAY.safe(REPLAY.node(node))
              expected_node = REPLAY.safe(event.get("node"))
              memory_status = "exact"
              if actual_node != expected_node:
                actual_without_filling = dict(actual_node)
                expected_without_filling = dict(expected_node or {})
                actual_without_filling.pop("filling", None)
                expected_without_filling.pop("filling", None)
                if actual_without_filling == expected_without_filling:
                  memory_status = "filling_mismatch"
                  REPLAY.warn(
                      f"{method_name} filling mismatch; using trace filling"
                  )
                else:
                  memory_status = "core_mismatch"
                  REPLAY.warn(
                      f"{method_name} node mismatch; using trace node\n"
                      f"actual={json.dumps(actual_node, ensure_ascii=True)[:1000]}\n"
                      f"expected={json.dumps(expected_node, ensure_ascii=True)[:1000]}"
                  )
              _canonicalize_memory_node(self, node, expected_node)
              REPLAY.perf(
                  type="memory",
                  event_id=event_id,
                  agent=REPLAY.current_agent(),
                  step=REPLAY.current_step(),
                  memory_kind=event.get("memory_kind"),
                  status=memory_status,
              )
            return node
          return replay_add

        setattr(cls, method_name, make_replay(method_name, original_method))

    return cls

  builtins.__build_class__ = replay_build_class


def _install_movement_replay_hook():
  original_open = builtins.open

  class ReplayMovementWriter:
    def __init__(self, wrapped, path):
      self.wrapped = wrapped
      self.path = path

    def __enter__(self):
      self.wrapped.__enter__()
      return self

    def __exit__(self, exc_type, exc, tb):
      return self.wrapped.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
      return getattr(self.wrapped, name)

    def write(self, data):
      result = self.wrapped.write(data)
      if data.strip():
        movement = json.loads(data)
        step = int(os.path.splitext(os.path.basename(self.path))[0])
        event_id = REPLAY.global_event_id("movement_commit", step=step)
        event = REPLAY.consume("movement_commit", event_id)
        if event:
          REPLAY.check_equal("movement_commit", movement, event.get("movement"))
      return result

  def replay_open(file, mode="r", *args, **kwargs):
    wrapped = original_open(file, mode, *args, **kwargs)
    if (
        "w" in mode
        and isinstance(file, str)
        and f"{os.sep}movement{os.sep}" in file
        and file.endswith(".json")
    ):
      return ReplayMovementWriter(wrapped, file)
    return wrapped

  builtins.open = replay_open


def _default_perf_path(sim):
  safe_sim = "".join(
      char if char.isalnum() or char in ("-", "_") else "_"
      for char in sim
  )
  return os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "perf",
      f"headless_replay_perf_{safe_sim}.jsonl",
  )


def _non_overwriting_path(path):
  if not os.path.exists(path):
    return path
  folder = os.path.dirname(path)
  stem, ext = os.path.splitext(os.path.basename(path))
  index = 2
  while True:
    candidate = os.path.join(folder, f"{stem}_{index}{ext}")
    if not os.path.exists(candidate):
      return candidate
    index += 1


def _latest_perf_path(sim):
  base = _default_perf_path(sim)
  folder = os.path.dirname(base) or "."
  stem, ext = os.path.splitext(os.path.basename(base))
  if not os.path.isdir(folder):
    return base
  candidates = []
  for name in os.listdir(folder):
    if name == f"{stem}{ext}" or (name.startswith(f"{stem}_") and name.endswith(ext)):
      candidates.append(os.path.join(folder, name))
  if not candidates:
    return base
  return max(candidates, key=os.path.getmtime)


def _default_report_path(sim, ext):
  safe_sim = "".join(
      char if char.isalnum() or char in ("-", "_") else "_"
      for char in sim
  )
  return os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "reports",
      f"headless_replay_report_{safe_sim}.{ext}",
  )


def _safe_name(name):
  return "".join(
      char if char.isalnum() or char in ("-", "_") else "_"
      for char in name
  )


def _default_run_dir(sim):
  return os.path.join(
      os.path.dirname(os.path.abspath(__file__)),
      "replay_runs",
      _safe_name(sim),
  )


def _write_run_config(path, config):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8") as outfile:
    json.dump(config, outfile, indent=2, ensure_ascii=True)
    outfile.write("\n")


def _percentile(values, pct):
  values = sorted(value for value in values if isinstance(value, (int, float)))
  if not values:
    return 0.0
  if len(values) == 1:
    return float(values[0])
  rank = (len(values) - 1) * pct / 100.0
  low = int(rank)
  high = min(low + 1, len(values) - 1)
  weight = rank - low
  return float(values[low] * (1 - weight) + values[high] * weight)


def _latency_summary(values):
  values = [value for value in values if isinstance(value, (int, float))]
  total = sum(values)
  return {
      "count": len(values),
      "total_ms": total,
      "avg_ms": total / len(values) if values else 0.0,
      "p50_ms": _percentile(values, 50),
      "p90_ms": _percentile(values, 90),
      "p95_ms": _percentile(values, 95),
      "p99_ms": _percentile(values, 99),
      "max_ms": max(values) if values else 0.0,
      "min_ms": min(values) if values else 0.0,
  }


def _counter_dict(counter):
  return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def _build_replay_report(trace_path, sim, perf_path, report_status, wall_ms):
  trace_events = REPLAY.events
  perf_events = list(REPLAY.perf_events)
  trace_counts = {}
  memory_kinds = {}
  random_fns = {}
  prompt_templates = {}
  prompt_record_by_sha = {}
  agents = set()
  steps = set()
  movement_chat = 0
  nonempty_retrieval = 0
  llm_request_by_id = {}
  embedding_request_by_id = {}
  prompt_records_by_context = {}

  for event in trace_events:
    event_type = event.get("type")
    trace_counts[event_type] = trace_counts.get(event_type, 0) + 1
    if event.get("agent"):
      agents.add(event.get("agent"))
    if event.get("step") is not None:
      steps.add(event.get("step"))
    if event_type == "memory_add":
      kind = event.get("memory_kind")
      memory_kinds[kind] = memory_kinds.get(kind, 0) + 1
    elif event_type == "random_result":
      fn = event.get("fn")
      random_fns[fn] = random_fns.get(fn, 0) + 1
    elif event_type == "prompt_built":
      template = event.get("prompt_template")
      prompt_templates[template] = prompt_templates.get(template, 0) + 1
      context_key = (event.get("agent"), event.get("step"))
      prompt_records_by_context.setdefault(context_key, []).append({
          "prompt_id": event.get("prompt_id"),
          "prompt_sha256": event.get("prompt_sha256"),
          "prompt_template": template,
          "prompt_template_sha256": event.get("prompt_template_sha256"),
      })
      prompt_sha = event.get("prompt_sha256")
      if prompt_sha:
        prompt_record_by_sha[prompt_sha] = {
            "prompt_id": event.get("prompt_id"),
            "prompt_sha256": prompt_sha,
            "prompt_template": template,
            "prompt_template_sha256": event.get("prompt_template_sha256"),
        }
    elif event_type == "movement_commit":
      for payload in event.get("movement", {}).get("persona", {}).values():
        if payload.get("chat") is not None:
          movement_chat += 1
    elif event_type == "retrieval_result":
      if event.get("focal_points"):
        nonempty_retrieval += 1
    elif event_type == "llm_request":
      llm_request_by_id[event.get("event_id")] = event
    elif event_type == "embedding_request":
      embedding_request_by_id[event.get("event_id")] = event

  by_type = {}
  by_agent = {}
  by_step = {}
  by_prompt_template = {}
  by_embedding_source = {}
  status_counts = {}
  token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
  token_seen = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
  prompt_context_index = {}

  for event in perf_events:
    event_type = event.get("type")
    status = event.get("status")
    if status:
      status_counts[status] = status_counts.get(status, 0) + 1
    latency = event.get("latency_ms")
    if event_type:
      by_type.setdefault(event_type, []).append(latency)
    agent = event.get("agent")
    if agent:
      by_agent.setdefault(agent, {}).setdefault(event_type, []).append(latency)
    step = event.get("step")
    if step is not None:
      by_step.setdefault(str(step), {}).setdefault(event_type, []).append(latency)

    if event_type == "llm":
      request = llm_request_by_id.get(event.get("event_id"), {})
      prompt_record = request.get("prompt_record") or (request.get("request") or {}).get("prompt_record") or {}
      if not prompt_record:
        messages = (request.get("request") or {}).get("messages") or []
        if messages and isinstance(messages, list):
          content = str(messages[0].get("content", ""))
          prompt_record = prompt_record_by_sha.get(hashlib.sha256(content.encode("utf-8")).hexdigest(), {})
          if not prompt_record and content.startswith('"""\n'):
            marker = '\n"""\nOutput the response to the prompt above in json.'
            end = content.find(marker)
            if end >= 0:
              inner = content[4:end]
              prompt_record = prompt_record_by_sha.get(hashlib.sha256(inner.encode("utf-8")).hexdigest(), {})
      if not prompt_record:
        context_key = (event.get("agent"), event.get("step"))
        records = prompt_records_by_context.get(context_key, [])
        idx = prompt_context_index.get(context_key, 0)
        if idx < len(records):
          prompt_record = records[idx]
          prompt_context_index[context_key] = idx + 1
      template = prompt_record.get("prompt_template") or "unknown"
      by_prompt_template.setdefault(template, []).append(latency)
      usage = event.get("usage") or {}
      for key in token_totals:
        value = usage.get(key)
        if isinstance(value, int):
          token_totals[key] += value
          token_seen[key] += 1
    elif event_type == "embedding":
      request = embedding_request_by_id.get(event.get("event_id"), {})
      input_value = (request.get("request") or {}).get("input")
      if isinstance(input_value, list):
        source = f"batch_{len(input_value)}"
      elif isinstance(input_value, str):
        source = "single_text"
      else:
        source = "unknown"
      by_embedding_source.setdefault(source, []).append(latency)

  type_summary = {key: _latency_summary(values) for key, values in by_type.items()}
  agent_summary = {
      agent: {key: _latency_summary(values) for key, values in groups.items()}
      for agent, groups in by_agent.items()
  }
  step_summary = {
      step: {key: _latency_summary(values) for key, values in groups.items()}
      for step, groups in by_step.items()
  }
  prompt_template_summary = {
      key: _latency_summary(values)
      for key, values in by_prompt_template.items()
  }
  embedding_source_summary = {
      key: _latency_summary(values)
      for key, values in by_embedding_source.items()
  }

  llm_ms = type_summary.get("llm", {}).get("total_ms", 0.0)
  embedding_ms = type_summary.get("embedding", {}).get("total_ms", 0.0)
  model_ms = llm_ms + embedding_ms
  step_agent_move_sum = {}
  step_agent_move_by_agent = {}
  step_model_sum = {}
  step_agent_model_sum = {}
  step_llm_sum = {}
  step_agent_llm_sum = {}
  for event in perf_events:
    latency = event.get("latency_ms")
    step = event.get("step")
    if not isinstance(latency, (int, float)) or step is None:
      continue
    agent = event.get("agent") or "unknown"
    if event.get("type") == "agent_move_total":
      step_agent_move_sum[step] = step_agent_move_sum.get(step, 0.0) + latency
      step_agent_move_by_agent[(step, agent)] = (
          step_agent_move_by_agent.get((step, agent), 0.0) + latency
      )
    if event.get("type") in ("llm", "embedding"):
      step_model_sum[step] = step_model_sum.get(step, 0.0) + latency
      step_agent_model_sum[(step, agent)] = step_agent_model_sum.get((step, agent), 0.0) + latency
    if event.get("type") == "llm":
      step_llm_sum[step] = step_llm_sum.get(step, 0.0) + latency
      step_agent_llm_sum[(step, agent)] = step_agent_llm_sum.get((step, agent), 0.0) + latency
  sequential_agent_move_ms = sum(step_agent_move_sum.values())
  agent_parallel_agent_move_ms = sum(
      max(
          value
          for (agent_step, _agent), value in step_agent_move_by_agent.items()
          if agent_step == step
      )
      for step in step_agent_move_sum
  )
  sequential_model_ms = sum(step_model_sum.values())
  agent_parallel_model_ms = sum(
      max(
          value
          for (agent_step, _agent), value in step_agent_model_sum.items()
          if agent_step == step
      )
      for step in step_model_sum
  )
  sequential_llm_ms = sum(step_llm_sum.values())
  agent_parallel_llm_ms = sum(
      max(
          value
          for (agent_step, _agent), value in step_agent_llm_sum.items()
          if agent_step == step
      )
      for step in step_llm_sum
  )
  step_non_model_sum = {}
  step_agent_non_model_sum = {}
  for key, move_latency in step_agent_move_by_agent.items():
    model_latency = step_agent_model_sum.get(key, 0.0)
    non_model_latency = max(0.0, move_latency - model_latency)
    step_agent_non_model_sum[key] = non_model_latency
    step = key[0]
    step_non_model_sum[step] = step_non_model_sum.get(step, 0.0) + non_model_latency
  sequential_non_model_ms = sum(step_non_model_sum.values())
  agent_parallel_non_model_ms = sum(
      max(
          value
          for (agent_step, _agent), value in step_agent_non_model_sum.items()
          if agent_step == step
      )
      for step in step_non_model_sum
  )

  replay_quality = {
      "retrieval_canonicalized": sum(
          1 for event in perf_events
          if event.get("type") == "retrieval" and event.get("status") == "canonicalized"
      ),
      "reflection_skipped_by_trace": sum(
          1 for event in perf_events
          if event.get("type") == "reflection" and event.get("status") == "skipped_by_trace"
      ),
      "memory_exact": sum(
          1 for event in perf_events
          if event.get("type") == "memory" and event.get("status") == "exact"
      ),
      "memory_filling_mismatch": sum(
          1 for event in perf_events
          if event.get("type") == "memory" and event.get("status") == "filling_mismatch"
      ),
      "memory_core_mismatch": sum(
          1 for event in perf_events
          if event.get("type") == "memory" and event.get("status") == "core_mismatch"
      ),
      "errors": sum(
          1 for event in perf_events
          if event.get("status") == "error" or event.get("type") == "replay_error"
      ),
  }

  simulation_steps = trace_counts.get("movement_commit", 0)
  agent_moves = trace_counts.get("agent_move_start", 0)
  report = {
      "schema_version": 1,
      "trace_path": trace_path,
      "replay_sim": sim,
      "perf_path": perf_path,
      "status": report_status,
      "overall": {
          "simulation_steps": simulation_steps,
          "agent_count": len(agents),
          "agent_moves": agent_moves,
          "wall_time_ms": wall_ms,
          "llm_total_ms": llm_ms,
          "embedding_total_ms": embedding_ms,
          "model_total_ms": model_ms,
          "agent_move_total_ms": sequential_agent_move_ms,
          "agent_move_non_model_total_ms": sequential_non_model_ms,
          "non_llm_total_ms": max(0.0, wall_ms - llm_ms),
          "overhead_excluding_llm_embedding_ms": max(0.0, wall_ms - model_ms),
          "llm_percentage_of_wall": (llm_ms / wall_ms * 100.0) if wall_ms else 0.0,
          "embedding_percentage_of_wall": (embedding_ms / wall_ms * 100.0) if wall_ms else 0.0,
          "model_percentage_of_wall": (model_ms / wall_ms * 100.0) if wall_ms else 0.0,
          "steps_per_second": (simulation_steps / (wall_ms / 1000.0)) if wall_ms else 0.0,
          "agent_moves_per_second": (agent_moves / (wall_ms / 1000.0)) if wall_ms else 0.0,
      },
      "latency_by_type": type_summary,
      "perf_event_counts": _counter_dict(REPLAY.perf_counts),
      "llm_by_prompt_template": prompt_template_summary,
      "latency_by_agent": agent_summary,
      "latency_by_step": step_summary,
      "embedding_by_source": embedding_source_summary,
      "token_totals": {
          **token_totals,
          "available_counts": token_seen,
      },
      "parallelism_estimate": {
          "sequential_agent_move_ms": sequential_agent_move_ms,
          "agent_parallel_agent_move_ms": agent_parallel_agent_move_ms,
          "sequential_model_ms": sequential_model_ms,
          "agent_parallel_model_ms": agent_parallel_model_ms,
          "sequential_llm_ms": sequential_llm_ms,
          "agent_parallel_llm_ms": agent_parallel_llm_ms,
          "sequential_non_model_ms": sequential_non_model_ms,
          "agent_parallel_non_model_ms": agent_parallel_non_model_ms,
          "estimated_agent_parallel_agent_move_speedup": (
              sequential_agent_move_ms / agent_parallel_agent_move_ms
              if agent_parallel_agent_move_ms else 0.0
          ),
          "estimated_agent_parallel_model_speedup": (
              sequential_model_ms / agent_parallel_model_ms
              if agent_parallel_model_ms else 0.0
          ),
          "estimated_agent_parallel_llm_speedup": (
              sequential_llm_ms / agent_parallel_llm_ms
              if agent_parallel_llm_ms else 0.0
          ),
          "estimated_agent_parallel_non_model_speedup": (
              sequential_non_model_ms / agent_parallel_non_model_ms
              if agent_parallel_non_model_ms else 0.0
          ),
          "avg_model_ms_per_step": (
              sequential_model_ms / simulation_steps
              if simulation_steps else 0.0
          ),
          "avg_agent_move_ms_per_step": (
              sequential_agent_move_ms / simulation_steps
              if simulation_steps else 0.0
          ),
          "avg_non_model_ms_per_step": (
              sequential_non_model_ms / simulation_steps
              if simulation_steps else 0.0
          ),
      },
      "trace_coverage": {
          "event_counts": _counter_dict(trace_counts),
          "memory_kinds": _counter_dict(memory_kinds),
          "random_fns": _counter_dict(random_fns),
          "prompt_templates": _counter_dict(prompt_templates),
          "movement_chat": movement_chat,
          "nonempty_retrieval": nonempty_retrieval,
          "unique_agents": sorted(agents),
          "unique_steps": len(steps),
      },
      "replay_quality": replay_quality,
      "perf_status_counts": _counter_dict(status_counts),
  }
  return report


def _write_report_json(report, path):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8") as outfile:
    json.dump(report, outfile, indent=2, ensure_ascii=True)


def _write_report_md(report, path):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  overall = report["overall"]
  quality = report["replay_quality"]
  parallel = report["parallelism_estimate"]
  with open(path, "w", encoding="utf-8") as outfile:
    outfile.write(f"# Headless Replay Report: {report['replay_sim']}\n\n")
    outfile.write(f"- status: `{report['status']}`\n")
    outfile.write(f"- trace: `{report['trace_path']}`\n")
    outfile.write(f"- perf: `{report['perf_path']}`\n\n")
    outfile.write("## Overall\n\n")
    for key in (
        "simulation_steps",
        "agent_count",
        "agent_moves",
        "wall_time_ms",
        "llm_total_ms",
        "embedding_total_ms",
        "model_total_ms",
        "agent_move_total_ms",
        "agent_move_non_model_total_ms",
        "overhead_excluding_llm_embedding_ms",
        "steps_per_second",
        "agent_moves_per_second",
    ):
      outfile.write(f"- {key}: {overall.get(key, 0):.3f}\n" if isinstance(overall.get(key), float) else f"- {key}: {overall.get(key)}\n")
    outfile.write("\n## Replay Quality\n\n")
    for key, value in quality.items():
      outfile.write(f"- {key}: {value}\n")
    outfile.write("\n## Parallelism Estimate\n\n")
    for key, value in parallel.items():
      if isinstance(value, (int, float)):
        outfile.write(f"- {key}: {value:.3f}\n")
      else:
        outfile.write(f"- {key}: {value}\n")
    outfile.write("\n## LLM By Prompt Template\n\n")
    for template, summary in sorted(
        report["llm_by_prompt_template"].items(),
        key=lambda item: item[1].get("total_ms", 0),
        reverse=True,
    ):
      outfile.write(
          f"- `{template}`: count={summary['count']}, "
          f"total_ms={summary['total_ms']:.3f}, avg_ms={summary['avg_ms']:.3f}, "
          f"p95_ms={summary['p95_ms']:.3f}\n"
      )
    outfile.write("\n## Trace Coverage\n\n")
    coverage = report["trace_coverage"]
    outfile.write(f"- movement_chat: {coverage['movement_chat']}\n")
    outfile.write(f"- nonempty_retrieval: {coverage['nonempty_retrieval']}\n")
    outfile.write(f"- memory_kinds: `{json.dumps(coverage['memory_kinds'], ensure_ascii=True)}`\n")
    outfile.write(f"- random_fns: `{json.dumps(coverage['random_fns'], ensure_ascii=True)}`\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--trace", required=True)
  parser.add_argument("--fork", help="Simulation folder to fork from.")
  parser.add_argument("--sim", help="New replay simulation folder.")
  parser.add_argument("--steps", type=int)
  parser.add_argument("--no-save", action="store_true")
  parser.add_argument("--non-strict", action="store_true")
  parser.add_argument("--perf", help="Performance JSONL output path.")
  parser.add_argument("--report", help="Replay statistics JSON output path.")
  parser.add_argument("--report-md", help="Replay statistics Markdown output path.")
  parser.add_argument(
      "--run-dir",
      help="Directory for replay outputs. Defaults to <this script dir>/replay_runs/<sim>.",
  )
  parser.add_argument("--report-only", action="store_true", help="Read trace/perf and write reports without running replay.")
  args = parser.parse_args()

  global REPLAY
  init_events = load_trace_for_defaults(args.trace)
  init_event = first_default_event(init_events, "simulation_init") or {}
  source_sim = init_event.get("sim_code") or "trace"
  sim = args.sim or f"replay_{source_sim}"
  run_dir = os.path.abspath(args.run_dir or _default_run_dir(sim))
  config_path = os.path.join(run_dir, "config.json")
  perf_default = os.path.join(run_dir, "perf.jsonl")
  report_default = os.path.join(run_dir, "report.json")
  report_md_default = os.path.join(run_dir, "report.md")
  perf_path = _non_overwriting_path(args.perf or perf_default)
  report_path = _non_overwriting_path(args.report or report_default)
  report_md_path = _non_overwriting_path(args.report_md or report_md_default)

  if args.report_only:
    existing_perf_path = args.perf or (
        os.path.join(run_dir, "perf.jsonl")
        if os.path.exists(os.path.join(run_dir, "perf.jsonl"))
        else _latest_perf_path(sim)
    )
    REPLAY = ReplayController(
        args.trace,
        strict=not args.non_strict,
        perf_path=None,
    )
    REPLAY.perf_path = existing_perf_path
    REPLAY.perf_events = load_trace_for_defaults(existing_perf_path)
    for event in REPLAY.perf_events:
      event_type = event.get("type")
      if event_type:
        REPLAY.perf_counts[event_type] = REPLAY.perf_counts.get(event_type, 0) + 1
      status = event.get("status")
      if status:
        REPLAY.perf_status_counts[status] = REPLAY.perf_status_counts.get(status, 0) + 1
      latency = event.get("latency_ms")
      if event_type and isinstance(latency, (int, float)):
        REPLAY.perf_latency_by_type[event_type] = (
            REPLAY.perf_latency_by_type.get(event_type, 0.0) + latency
        )
    total_event = next(
        (event for event in reversed(REPLAY.perf_events) if event.get("type") == "replay_total"),
        None,
    )
    wall_ms = total_event.get("latency_ms", 0.0) if total_event else 0.0
    status = total_event.get("status", "unknown") if total_event else "unknown"
    report = _build_replay_report(args.trace, sim, existing_perf_path, status, wall_ms)
    _write_report_json(report, report_path)
    _write_report_md(report, report_md_path)
    _write_run_config(config_path, {
        "mode": "report_only",
        "trace": args.trace,
        "source_sim": source_sim,
        "fork_sim": args.fork or init_event.get("fork_sim_code"),
        "replay_sim": sim,
        "status": status,
        "perf": existing_perf_path,
        "report": report_path,
        "report_md": report_md_path,
        "figures": os.path.join(run_dir, "figures"),
        "storage": os.path.join(os.environ.get("FS_STORAGE", "../environment/frontend_server/storage"), sim),
    })
    print(f"[headless replay] report: {report_path}")
    print(f"[headless replay] report_md: {report_md_path}")
    print(f"[headless replay] run_dir: {run_dir}")
    return

  fork = args.fork or init_event.get("fork_sim_code")
  steps = args.steps
  if steps is None:
    run_events = [event for event in init_events if event.get("type") == "run_start"]
    steps = run_events[0].get("requested_steps") if run_events else None
  if not fork or not steps:
    raise ReplayMismatch("Need --fork/trace simulation_init and --steps/run_start")

  os.makedirs(run_dir, exist_ok=True)

  REPLAY = ReplayController(
      args.trace,
      strict=not args.non_strict,
      perf_path=perf_path,
  )

  sglang_patch._install_logging()
  sglang_patch._install_default_utils_if_missing()
  sglang_patch._install_fast_fork_copy()
  sglang_patch._install_reverie_server_init_hook()

  _install_import_replay_hooks()
  _install_class_replay_hooks()
  _install_movement_replay_hook()
  _install_openai_replay_hooks()
  _install_random_replay_hooks()

  start_time_ns = time.perf_counter_ns()
  replay_status = "ok"
  replay_returncode = 0
  try:
    from reverie import ReverieServer

    print(f"[headless replay] trace={args.trace}")
    print(f"[headless replay] fork={fork} sim={sim} steps={steps}")
    print(f"[headless replay] run_dir={run_dir}")
    server = ReverieServer(fork, sim)
    server.start_server(int(steps))
    if not args.no_save:
      _compact_server_memories(server)
      server.save()
    print(f"[headless replay] complete: {sim}")
    print(f"[headless replay] perf: {REPLAY.perf_path}")
  except Exception:
    replay_status = "error"
    replay_returncode = 1
    REPLAY.perf(
        type="replay_error",
        status="error",
        error=traceback.format_exc(),
    )
    raise
  finally:
    end_time_ns = time.perf_counter_ns()
    wall_ms = (end_time_ns - start_time_ns) / 1_000_000
    llm_ms = REPLAY.perf_latency_by_type.get("llm", 0.0)
    embedding_ms = REPLAY.perf_latency_by_type.get("embedding", 0.0)
    REPLAY.perf(
        type="replay_total",
        status=replay_status,
        returncode=replay_returncode,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=wall_ms,
        llm_total_ms=llm_ms,
        embedding_total_ms=embedding_ms,
        non_llm_total_ms=max(0.0, wall_ms - llm_ms),
        overhead_excluding_llm_embedding_ms=max(0.0, wall_ms - llm_ms - embedding_ms),
        event_counts=REPLAY.perf_counts,
        status_counts=REPLAY.perf_status_counts,
    )
    try:
      report = _build_replay_report(
          args.trace,
          sim,
          REPLAY.perf_path,
          replay_status,
          wall_ms,
      )
      _write_report_json(report, report_path)
      _write_report_md(report, report_md_path)
      _write_run_config(config_path, {
          "mode": "replay",
          "trace": args.trace,
          "source_sim": source_sim,
          "fork_sim": fork,
          "replay_sim": sim,
          "steps": steps,
          "status": replay_status,
          "returncode": replay_returncode,
          "perf": REPLAY.perf_path,
          "report": report_path,
          "report_md": report_md_path,
          "figures": os.path.join(run_dir, "figures"),
          "storage": os.path.join(os.environ.get("FS_STORAGE", "../environment/frontend_server/storage"), sim),
      })
      print(f"[headless replay] report: {report_path}")
      print(f"[headless replay] report_md: {report_md_path}")
      print(f"[headless replay] run_dir: {run_dir}")
    except Exception:
      REPLAY.perf(
          type="report_error",
          status="error",
          error=traceback.format_exc(),
      )
    REPLAY.close()


if __name__ == "__main__":
  main()
