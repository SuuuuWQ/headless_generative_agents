"""
Run the headless Reverie copy with the SGLang patch and record a replay trace.

Usage, from headless_genagents:
  python headless_trace_runner.py --fork test3 --sim headless_trace_test
  Enter option: run 20
  Enter option: run 100
  Enter option: fin

For one-shot smoke tests:
  python headless_trace_runner.py --fork test3 --sim headless_trace_test --steps 20

The trace is JSONL: one event per line. By default it is written to
traces/trace_<sim>.jsonl and overwritten for every new run. Set
REVERIE_TRACE_FILE to choose another path.
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
import traceback

import openai

import sglang_openai_patch as sglang_patch


TRACE_DIR = os.environ.get("REVERIE_TRACE_DIR", "traces")
TRACE_FILE = os.environ.get("REVERIE_TRACE_FILE")
TRACE_RECORD_FULL_PROMPT = (
    os.environ.get("TRACE_RECORD_FULL_PROMPT", "False").lower() == "true"
)
TRACE_RECORD_EMBEDDINGS = (
    os.environ.get("TRACE_RECORD_EMBEDDINGS", "False").lower() == "true"
)


class TraceRecorder:
  def __init__(self, path=None):
    self.path = path
    self.lock = threading.Lock()
    self.seq = 0
    self.prompt_by_text = {}
    self.context = threading.local()
    self.sim_base_step = None
    self.sim_base_time = None
    self.sec_per_step = None
    self.is_running = False
    self.event_counters = {}
    self.file = None
    self.buffered_events = []
    if path:
      self.open_path(path)

  def open_path(self, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    self.path = path
    self.file = open(path, "w", encoding="utf-8", buffering=1)
    for event in self.buffered_events:
      self.file.write(json.dumps(event, ensure_ascii=True) + "\n")
    self.buffered_events = []

  def configure_for_sim(self, sim_code):
    if self.file:
      return
    path = TRACE_FILE
    if not path:
      safe_sim_code = "".join(
          char if char.isalnum() or char in ("-", "_") else "_"
          for char in sim_code
      )
      path = os.path.join(TRACE_DIR, f"trace_{safe_sim_code}.jsonl")
    self.open_path(path)

  def close(self):
    if self.file:
      self.file.close()

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

  def emit(self, event_type, **fields):
    with self.lock:
      self.seq += 1
      event = {
          "seq": self.seq,
          "type": event_type,
      }
      event.update({key: self.safe(val) for key, val in fields.items()})
      if self.file:
        self.file.write(json.dumps(event, ensure_ascii=True) + "\n")
      else:
        self.buffered_events.append(event)

  def sha256_file(self, path):
    try:
      with open(path, "rb") as infile:
        return hashlib.sha256(infile.read()).hexdigest()
    except Exception:
      return None

  def sha256_text(self, text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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

  def set_sim_clock(self, step, curr_time, sec_per_step):
    self.sim_base_step = step
    self.sim_base_time = curr_time
    self.sec_per_step = sec_per_step

  def start_run(self):
    self.is_running = True

  def end_run(self):
    self.is_running = False

  def estimate_step(self, curr_time):
    if (
        self.sim_base_step is None
        or self.sim_base_time is None
        or not self.sec_per_step
        or not curr_time
    ):
      return None
    try:
      elapsed = (curr_time - self.sim_base_time).total_seconds()
      return int(self.sim_base_step + (elapsed / self.sec_per_step))
    except Exception:
      return None

  def slug(self, value, max_len=48):
    if value is None:
      return "none"
    value = str(value)
    chars = []
    for char in value:
      if char.isalnum():
        chars.append(char)
      elif char in ("-", "_", "."):
        chars.append(char)
      elif char in ("/", "\\"):
        chars.append(".")
      else:
        chars.append("_")
    slug = "".join(chars).strip("_")
    return (slug or "none")[:max_len]

  def event_id(self, kind, step=None, agent=None, label=None):
    if step is None:
      step = self.current_step()
    if agent is None:
      agent = self.current_agent()
    key = (step, agent, kind)
    local_index = self.event_counters.get(key, 0) + 1
    self.event_counters[key] = local_index
    parts = [
        "step" if step is None else str(step),
        "global" if agent is None else self.slug(agent),
        self.slug(kind),
        str(local_index),
    ]
    if label:
      parts.append(self.slug(label))
    return "|".join(parts)

  def global_event_id(self, kind, step=None, label=None):
    if step is None:
      step = self.current_step()
    key = (step, "global", kind)
    local_index = self.event_counters.get(key, 0) + 1
    self.event_counters[key] = local_index
    parts = [
        "step" if step is None else str(step),
        "global",
        self.slug(kind),
        str(local_index),
    ]
    if label:
      parts.append(self.slug(label))
    return "|".join(parts)

  def remember_local_event_id(self, key, event_id):
    local_ids = getattr(self.context, "local_event_ids", None)
    if local_ids is None:
      local_ids = {}
      self.context.local_event_ids = local_ids
    local_ids[key] = event_id

  def local_event_id(self, key):
    return getattr(self.context, "local_event_ids", {}).get(key)

  def record_prompt(self, prompt, prompt_input, prompt_template):
    prompt_id = f"prompt_{len(self.prompt_by_text) + 1}"
    event_id = self.event_id("prompt", label=os.path.basename(prompt_template))
    record = {
        "event_id": event_id,
        "prompt_id": prompt_id,
        "prompt_sha256": self.sha256_text(prompt),
        "prompt_template": prompt_template,
        "prompt_template_sha256": self.sha256_file(prompt_template),
        "prompt_input": prompt_input,
    }
    if TRACE_RECORD_FULL_PROMPT:
      record["prompt"] = prompt
    self.prompt_by_text[prompt] = record
    self.emit(
        "prompt_built",
        agent=self.current_agent(),
        step=self.current_step(),
        **record,
    )
    return record

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

  def scratch_summary(self, scratch):
    return {
        "curr_tile": getattr(scratch, "curr_tile", None),
        "curr_time": getattr(scratch, "curr_time", None),
        "act_address": getattr(scratch, "act_address", None),
        "act_event": getattr(scratch, "act_event", None),
        "act_obj_event": getattr(scratch, "act_obj_event", None),
        "act_description": getattr(scratch, "act_description", None),
        "act_pronunciatio": getattr(scratch, "act_pronunciatio", None),
        "planned_path": getattr(scratch, "planned_path", None),
        "chat": getattr(scratch, "chat", None),
    }


TRACE = TraceRecorder(TRACE_FILE)


def _completion_texts(response):
  try:
    return [choice.get("text", "") for choice in response.get("choices", [])]
  except Exception:
    return []


def _chat_texts(response):
  try:
    return [
        choice.get("message", {}).get("content", "")
        for choice in response.get("choices", [])
    ]
  except Exception:
    return []


def _embedding_summary(response):
  data = response.get("data", []) if isinstance(response, dict) else []
  summary = {
      "count": len(data),
      "dimensions": [],
  }
  for item in data:
    embedding = item.get("embedding")
    summary["dimensions"].append(len(embedding) if embedding is not None else None)
  if TRACE_RECORD_EMBEDDINGS:
    summary["data"] = data
  return summary


def _install_openai_trace_hooks():
  original_completion_create = openai.Completion.create
  original_chat_create = openai.ChatCompletion.create
  original_embedding_create = openai.Embedding.create

  def traced_completion_create(*args, **kwargs):
    prompt = kwargs.get("prompt")
    if prompt is None and args:
      prompt = args[0]
    call_id = f"llm_{TRACE.seq + 1}"
    prompt_record = TRACE.compact_prompt_record(prompt)
    event_id = TRACE.event_id(
        "llm",
        label=(prompt_record or {}).get("prompt_template") or "completion",
    )
    TRACE.emit(
        "llm_request",
        event_id=event_id,
        call_id=call_id,
        api="completion",
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        prompt_record=prompt_record,
        request={
            "model": kwargs.get("model"),
            "engine": kwargs.get("engine"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop": kwargs.get("stop"),
            "stream": kwargs.get("stream"),
            "prompt": None if prompt_record else prompt,
        },
    )
    try:
      response = original_completion_create(*args, **kwargs)
    except Exception:
      TRACE.emit(
          "llm_response",
          event_id=event_id,
          call_id=call_id,
          api="completion",
          agent=TRACE.current_agent(),
          step=TRACE.current_step(),
          status="error",
          error=traceback.format_exc(),
          canonical_texts=[],
      )
      raise
    TRACE.emit(
        "llm_response",
        event_id=event_id,
        call_id=call_id,
        api="completion",
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        status="ok",
        canonical_texts=_completion_texts(response),
    )
    return response

  def traced_chat_create(*args, **kwargs):
    call_id = f"llm_{TRACE.seq + 1}"
    messages = kwargs.get("messages") or []
    prompt = None
    label = "chat"
    if messages and isinstance(messages, list):
      prompt = str(messages[0].get("content", ""))
      label = hashlib.sha256(
          prompt.encode("utf-8")
      ).hexdigest()[:12]
    prompt_record = TRACE.compact_prompt_record(prompt)
    if prompt_record:
      label = prompt_record.get("prompt_template") or label
    event_id = TRACE.event_id("llm", label=label)
    TRACE.emit(
        "llm_request",
        event_id=event_id,
        call_id=call_id,
        api="chat",
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        prompt_record=prompt_record,
        request={
            "model": kwargs.get("model"),
            "messages": None if prompt_record else kwargs.get("messages"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p"),
            "stop": kwargs.get("stop"),
            "stream": kwargs.get("stream"),
        },
    )
    try:
      response = original_chat_create(*args, **kwargs)
    except Exception:
      TRACE.emit(
          "llm_response",
          event_id=event_id,
          call_id=call_id,
          api="chat",
          agent=TRACE.current_agent(),
          step=TRACE.current_step(),
          status="error",
          error=traceback.format_exc(),
          canonical_texts=[],
      )
      raise
    TRACE.emit(
        "llm_response",
        event_id=event_id,
        call_id=call_id,
        api="chat",
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        status="ok",
        canonical_texts=_chat_texts(response),
    )
    return response

  def traced_embedding_create(*args, **kwargs):
    input_text = kwargs.get("input")
    if input_text is None and args:
      input_text = args[0]
    call_id = f"embedding_{TRACE.seq + 1}"
    event_id = TRACE.event_id("embedding")
    TRACE.emit(
        "embedding_request",
        event_id=event_id,
        call_id=call_id,
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        request={
            "model": kwargs.get("model"),
            "input": input_text,
        },
    )
    try:
      response = original_embedding_create(*args, **kwargs)
    except Exception:
      TRACE.emit(
          "embedding_response",
          event_id=event_id,
          call_id=call_id,
          agent=TRACE.current_agent(),
          step=TRACE.current_step(),
          status="error",
          error=traceback.format_exc(),
          canonical_summary={"count": 0, "dimensions": []},
      )
      raise
    TRACE.emit(
        "embedding_response",
        event_id=event_id,
        call_id=call_id,
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        status="ok",
        canonical_summary=_embedding_summary(response),
    )
    return response

  openai.Completion.create = staticmethod(traced_completion_create)
  openai.ChatCompletion.create = staticmethod(traced_chat_create)
  openai.Embedding.create = staticmethod(traced_embedding_create)


def _install_random_trace_hooks():
  original_choice = random.choice
  original_choices = random.choices
  original_randint = random.randint
  original_sample = random.sample

  def traced_choice(seq):
    result = original_choice(seq)
    TRACE.emit(
        "random_result",
        event_id=TRACE.event_id("random", label="choice"),
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        fn="choice",
        args={"seq": list(seq) if not isinstance(seq, range) else repr(seq)},
        result=result,
    )
    return result

  def traced_choices(population, weights=None, cum_weights=None, k=1):
    result = original_choices(
        population, weights=weights, cum_weights=cum_weights, k=k
    )
    TRACE.emit(
        "random_result",
        event_id=TRACE.event_id("random", label="choices"),
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        fn="choices",
        args={
            "population": list(population),
            "weights": weights,
            "cum_weights": cum_weights,
            "k": k,
        },
        result=result,
    )
    return result

  def traced_randint(a, b):
    result = original_randint(a, b)
    TRACE.emit(
        "random_result",
        event_id=TRACE.event_id("random", label="randint"),
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        fn="randint",
        args={"a": a, "b": b},
        result=result,
    )
    return result

  def traced_sample(population, k, counts=None):
    if counts is None:
      result = original_sample(population, k)
    else:
      result = original_sample(population, k, counts=counts)
    TRACE.emit(
        "random_result",
        event_id=TRACE.event_id("random", label="sample"),
        agent=TRACE.current_agent(),
        step=TRACE.current_step(),
        fn="sample",
        args={
            "population": list(population),
            "k": k,
            "counts": counts,
        },
        result=result,
    )
    return result

  random.choice = traced_choice
  random.choices = traced_choices
  random.randint = traced_randint
  random.sample = traced_sample


def _install_import_trace_hooks():
  original_import = builtins.__import__

  def traced_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)
    if name == "persona.prompt_template.gpt_structure":
      original_generate_prompt = getattr(module, "generate_prompt", None)
      if original_generate_prompt and not getattr(
          original_generate_prompt, "_trace_wrapped", False
      ):

        def traced_generate_prompt(curr_input, prompt_lib_file):
          prompt = original_generate_prompt(curr_input, prompt_lib_file)
          TRACE.record_prompt(prompt, curr_input, prompt_lib_file)
          return prompt

        traced_generate_prompt._trace_wrapped = True
        module.generate_prompt = traced_generate_prompt
    return module

  builtins.__import__ = traced_import


def _install_class_trace_hooks():
  original_build_class = builtins.__build_class__

  def traced_build_class(func, name, *args, **kwargs):
    cls = original_build_class(func, name, *args, **kwargs)

    if name == "ReverieServer":
      original_init = cls.__init__
      original_start_server = cls.start_server

      def traced_init(self, fork_sim_code, sim_code):
        original_init(self, fork_sim_code, sim_code)
        TRACE.configure_for_sim(sim_code)
        TRACE.set_sim_clock(
            getattr(self, "step", None),
            getattr(self, "curr_time", None),
            getattr(self, "sec_per_step", None),
        )
        TRACE.emit(
            "simulation_init",
            event_id=TRACE.global_event_id(
                "simulation",
                step=getattr(self, "step", None),
                label="init",
            ),
            fork_sim_code=fork_sim_code,
            sim_code=sim_code,
            step=getattr(self, "step", None),
            curr_time=getattr(self, "curr_time", None),
            sec_per_step=getattr(self, "sec_per_step", None),
            server_sleep=getattr(self, "server_sleep", None),
            personas=list(getattr(self, "personas", {}).keys()),
            personas_tile=getattr(self, "personas_tile", None),
        )

      def traced_start_server(self, int_counter):
        TRACE.start_run()
        TRACE.emit(
            "run_start",
            event_id=TRACE.global_event_id(
                "run",
                step=getattr(self, "step", None),
                label="start",
            ),
            sim_code=getattr(self, "sim_code", None),
            start_step=getattr(self, "step", None),
            curr_time=getattr(self, "curr_time", None),
            requested_steps=int_counter,
            agent_order=list(getattr(self, "personas", {}).keys()),
        )
        try:
          return original_start_server(self, int_counter)
        finally:
          TRACE.emit(
              "run_end",
              event_id=TRACE.global_event_id(
                  "run",
                  step=getattr(self, "step", None),
                  label="end",
              ),
              sim_code=getattr(self, "sim_code", None),
              end_step=getattr(self, "step", None),
              curr_time=getattr(self, "curr_time", None),
          )
          TRACE.end_run()

      cls.__init__ = traced_init
      cls.start_server = traced_start_server

    elif name == "Persona":
      original_move = cls.move
      original_retrieve = cls.retrieve

      def traced_move(self, maze, personas, curr_tile, curr_time):
        curr_step = TRACE.estimate_step(curr_time)
        TRACE.set_agent_context(getattr(self, "name", None), curr_step)
        move_event_id = TRACE.event_id("agent_move")
        TRACE.remember_local_event_id("agent_move", move_event_id)
        TRACE.emit(
            "agent_move_start",
            event_id=move_event_id,
            agent=getattr(self, "name", None),
            step=curr_step,
            curr_tile=curr_tile,
            curr_time=curr_time,
            scratch_before=TRACE.scratch_summary(self.scratch),
        )
        try:
          output = original_move(self, maze, personas, curr_tile, curr_time)
          TRACE.emit(
              "agent_move_end",
              event_id=move_event_id,
              agent=getattr(self, "name", None),
              curr_time=curr_time,
              output={
                  "next_tile": output[0],
                  "pronunciatio": output[1],
                  "description": output[2],
              },
              scratch_after=TRACE.scratch_summary(self.scratch),
          )
          return output
        finally:
          TRACE.clear_agent_context()

      def traced_retrieve(self, perceived):
        retrieved = original_retrieve(self, perceived)
        TRACE.emit(
            "retrieval_result",
            event_id=TRACE.event_id("retrieval"),
            agent=getattr(self, "name", None),
            step=TRACE.current_step(),
            focal_points={
                focal_point: {
                    "curr_event": TRACE.node(groups.get("curr_event")),
                    "events": [
                        node.node_id for node in groups.get("events", [])
                    ],
                    "thoughts": [
                        node.node_id for node in groups.get("thoughts", [])
                    ],
                }
                for focal_point, groups in retrieved.items()
            },
        )
        return retrieved

      cls.move = traced_move
      cls.retrieve = traced_retrieve

    elif name == "AssociativeMemory":
      for method_name in ("add_event", "add_thought", "add_chat"):
        original_method = getattr(cls, method_name)

        def make_traced(method_name, original_method):
          def traced_add(self, *method_args, **method_kwargs):
            node = original_method(self, *method_args, **method_kwargs)
            if TRACE.is_running:
              TRACE.emit(
                  "memory_add",
                  event_id=TRACE.event_id(
                      "memory",
                      label=method_name.replace("add_", ""),
                  ),
                  agent=TRACE.current_agent(),
                  step=TRACE.current_step(),
                  memory_kind=method_name.replace("add_", ""),
                  node=TRACE.node(node),
              )
            return node
          return traced_add

        setattr(cls, method_name, make_traced(method_name, original_method))

    return cls

  builtins.__build_class__ = traced_build_class


def _install_movement_trace_hook():
  original_open = builtins.open

  class TraceMovementWriter:
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
      try:
        if data.strip():
          movement = json.loads(data)
          step = int(os.path.splitext(os.path.basename(self.path))[0])
          TRACE.emit(
              "movement_commit",
              event_id=TRACE.global_event_id(
                  "movement_commit",
                  step=step,
              ),
              step=step,
              path=self.path,
              movement=movement,
          )
      except Exception:
        TRACE.emit("trace_error", error=traceback.format_exc(), path=self.path)
      return result

  def traced_open(file, mode="r", *args, **kwargs):
    wrapped = original_open(file, mode, *args, **kwargs)
    if (
        "w" in mode
        and isinstance(file, str)
        and f"{os.sep}movement{os.sep}" in file
        and file.endswith(".json")
    ):
      return TraceMovementWriter(wrapped, file)
    return wrapped

  builtins.open = traced_open


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--fork", help="Simulation folder to fork from.")
  parser.add_argument("--sim", help="New simulation folder name.")
  parser.add_argument("--steps", type=int)
  parser.add_argument("--no-save", action="store_true")
  args = parser.parse_args()

  sglang_patch._install_logging()
  sglang_patch._install_default_utils_if_missing()
  sglang_patch._install_fast_fork_copy()
  sglang_patch._install_reverie_server_init_hook()

  _install_import_trace_hooks()
  _install_class_trace_hooks()
  _install_movement_trace_hook()
  _install_openai_trace_hooks()
  _install_random_trace_hooks()

  TRACE.emit(
      "trace_session_start",
      event_id=TRACE.global_event_id("trace_session", label="start"),
      trace_file=TRACE_FILE,
      trace_dir=TRACE_DIR,
      argv=sys.argv,
      sglang_model=sglang_patch.SGLANG_MODEL,
      sglang_api_base=sglang_patch.SGLANG_API_BASE,
      sglang_embedding_model=sglang_patch.SGLANG_EMBEDDING_MODEL,
      sglang_embedding_api_base=sglang_patch.SGLANG_EMBEDDING_API_BASE,
      record_full_prompt=TRACE_RECORD_FULL_PROMPT,
      record_embeddings=TRACE_RECORD_EMBEDDINGS,
  )

  try:
    from reverie import ReverieServer

    fork = args.fork or input("Enter the name of the forked simulation: ").strip()
    sim = args.sim or input("Enter the name of the new simulation: ").strip()

    server = ReverieServer(fork, sim)
    if args.steps is None:
      server.open_server()
      print(f"Headless trace session complete: {sim}")
    else:
      server.start_server(args.steps)
      if not args.no_save:
        server.save()
      print(f"Headless trace run complete: {sim}, steps={args.steps}")
  except Exception:
    TRACE.emit(
        "trace_session_error",
        event_id=TRACE.global_event_id("trace_session", label="error"),
        error=traceback.format_exc(),
    )
    raise
  finally:
    TRACE.emit(
        "trace_session_end",
        event_id=TRACE.global_event_id("trace_session", label="end"),
    )
    TRACE.close()


if __name__ == "__main__":
  main()
