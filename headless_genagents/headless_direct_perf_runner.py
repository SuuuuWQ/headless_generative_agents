"""
Run the headless GA directly and record timeline-friendly performance events.

Example:
  python headless_direct_perf_runner.py \
    --fork July1_the_ville_isabella_maria_klaus-step-3-14 \
    --sim direct_perf_100_a \
    --steps 100
"""
import argparse
import builtins
import hashlib
import json
import os
import sys
import time
import traceback

import openai

import sglang_openai_patch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(SCRIPT_DIR, "direct_run_perf_runs")
DIRECT = None


class PerfRecorder:
  def __init__(self, path):
    self.path = path
    self.seq = 0
    self.events = []
    self.counts = {}
    self.status_counts = {}
    self.latency_by_type = {}
    self.file = None
    if path:
      os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
      self.file = open(path, "w", encoding="utf-8", buffering=1)

  def close(self):
    if self.file:
      self.file.close()
      self.file = None

  def record(self, **event):
    self.seq += 1
    event = {"seq": self.seq, **event}
    event_type = event.get("type")
    status = event.get("status")
    latency = event.get("latency_ms")
    if event_type:
      self.counts[event_type] = self.counts.get(event_type, 0) + 1
      if isinstance(latency, (int, float)):
        self.latency_by_type[event_type] = (
            self.latency_by_type.get(event_type, 0.0) + latency
        )
    if status:
      self.status_counts[status] = self.status_counts.get(status, 0) + 1
    self.events.append(event)
    if self.file:
      self.file.write(json.dumps(event, ensure_ascii=True) + "\n")

  def timed(self, event_type, **fields):
    return _TimedEvent(self, event_type, fields)

  def summary(self):
    return {
        "counts": dict(sorted(self.counts.items())),
        "status_counts": dict(sorted(self.status_counts.items())),
        "latency_by_type": dict(sorted(self.latency_by_type.items())),
    }


class _TimedEvent:
  def __init__(self, recorder, event_type, fields):
    self.recorder = recorder
    self.event_type = event_type
    self.fields = fields
    self.started = None
    self.status = "ok"
    self.error = None

  def __enter__(self):
    self.started = time.perf_counter()
    self.start_time_ns = time.time_ns()
    return self

  def fail(self, error):
    self.status = "error"
    self.error = repr(error)

  def __exit__(self, exc_type, exc, tb):
    if exc is not None:
      self.fail(exc)
    elapsed_ms = (time.perf_counter() - self.started) * 1000.0
    end_time_ns = time.time_ns()
    self.recorder.record(
        type=self.event_type,
        status=self.status,
        error=self.error,
        start_time_ns=self.start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=elapsed_ms,
        **self.fields,
    )
    return False


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


class DirectPerfController:
  def __init__(self, perf_path):
    self.perf = PerfRecorder(perf_path)
    self.agent = None
    self.step = None
    self.server = None
    self.move_windows = {}
    self.prompt_by_text = {}

  def close(self):
    self.perf.close()

  def set_server(self, server):
    self.server = server

  def current_step(self):
    if self.server is None:
      return None
    return getattr(self.server, "step", None)

  def current_agent(self):
    return self.agent

  def set_agent_context(self, agent, step):
    self.agent = agent
    self.step = step

  def clear_agent_context(self):
    self.agent = None
    self.step = None

  def record_move_window(self, step, agent, start_time_ns, end_time_ns, latency_ms):
    self.move_windows.setdefault(step, []).append(
        {
            "agent": agent,
            "start_time_ns": start_time_ns,
            "end_time_ns": end_time_ns,
            "latency_ms": latency_ms,
        }
    )

  def flush_step(self, step):
    rows = self.move_windows.pop(step, None)
    if not rows:
      return
    start_time_ns = min(row["start_time_ns"] for row in rows)
    end_time_ns = max(row["end_time_ns"] for row in rows)
    sum_agent_ms = sum(row["latency_ms"] for row in rows)
    max_agent_ms = max(row["latency_ms"] for row in rows)
    self.perf.record(
        type="direct_step",
        step=step,
        status="ok",
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=(end_time_ns - start_time_ns) / 1_000_000,
        agent_count=len(rows),
        sum_agent_ms=sum_agent_ms,
        max_agent_ms=max_agent_ms,
    )

  def sha256_text(self, text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

  def sha256_file(self, path):
    try:
      with open(path, "rb") as infile:
        return hashlib.sha256(infile.read()).hexdigest()
    except OSError:
      return None

  def record_prompt(self, prompt, prompt_input, prompt_template):
    record = {
        "prompt_sha256": self.sha256_text(prompt),
        "prompt_template": prompt_template,
        "prompt_template_sha256": self.sha256_file(prompt_template),
    }
    self.prompt_by_text[prompt] = record
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
    return dict(record)


def _chat_prompt_from_kwargs(kwargs):
  messages = kwargs.get("messages") or []
  if messages and isinstance(messages, list):
    return str(messages[0].get("content", ""))
  return None


def _install_prompt_perf_hooks():
  original_import = builtins.__import__

  def wrap_generate_prompt(module):
    original_generate_prompt = getattr(module, "generate_prompt", None)
    if not original_generate_prompt or getattr(
        original_generate_prompt, "_direct_perf_wrapped", False
    ):
      return

    def direct_generate_prompt(curr_input, prompt_lib_file):
      prompt = original_generate_prompt(curr_input, prompt_lib_file)
      if DIRECT is not None:
        DIRECT.record_prompt(prompt, curr_input, prompt_lib_file)
      return prompt

    direct_generate_prompt._direct_perf_wrapped = True
    if getattr(original_generate_prompt, "_trace_wrapped", False):
      direct_generate_prompt._trace_wrapped = True
    if getattr(original_generate_prompt, "_replay_wrapped", False):
      direct_generate_prompt._replay_wrapped = True
    module.generate_prompt = direct_generate_prompt

  def direct_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)
    if name == "persona.prompt_template.gpt_structure":
      wrap_generate_prompt(module)
    return module

  builtins.__import__ = direct_import
  loaded = sys.modules.get("persona.prompt_template.gpt_structure")
  if loaded:
    wrap_generate_prompt(loaded)


def _install_openai_perf_hooks():
  original_completion_create = openai.Completion.create
  original_chat_create = openai.ChatCompletion.create
  original_embedding_create = openai.Embedding.create

  def direct_completion_create(*args, **kwargs):
    prompt = kwargs.get("prompt")
    if prompt is None and args:
      prompt = args[0]
    prompt_record = DIRECT.compact_prompt_record(prompt)
    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    error = None
    sglang_openai_patch.clear_last_llm_payload()
    try:
      response = original_completion_create(*args, **kwargs)
    except Exception as exc:
      response = None
      status = "error"
      error = repr(exc)
      raise
    finally:
      end_time_ns = time.time_ns()
      DIRECT.perf.record(
          type="worker_llm",
          api="completion",
          agent=DIRECT.current_agent(),
          step=DIRECT.step,
          status=status,
          error=error,
          start_time_ns=start_time_ns,
          end_time_ns=end_time_ns,
          latency_ms=(end_time_ns - start_time_ns) / 1_000_000,
          prompt_record=prompt_record,
          response_texts=_completion_texts(response),
          effective_request=sglang_openai_patch.get_last_llm_payload(),
      )
    return response

  def direct_chat_create(*args, **kwargs):
    prompt = _chat_prompt_from_kwargs(kwargs)
    prompt_record = DIRECT.compact_prompt_record(prompt)
    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    error = None
    sglang_openai_patch.clear_last_llm_payload()
    try:
      response = original_chat_create(*args, **kwargs)
    except Exception as exc:
      response = None
      status = "error"
      error = repr(exc)
      raise
    finally:
      end_time_ns = time.time_ns()
      DIRECT.perf.record(
          type="worker_llm",
          api="chat",
          agent=DIRECT.current_agent(),
          step=DIRECT.step,
          status=status,
          error=error,
          start_time_ns=start_time_ns,
          end_time_ns=end_time_ns,
          latency_ms=(end_time_ns - start_time_ns) / 1_000_000,
          prompt_record=prompt_record,
          response_texts=_chat_texts(response),
          effective_request=sglang_openai_patch.get_last_llm_payload(),
      )
    return response

  def direct_embedding_create(*args, **kwargs):
    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    error = None
    try:
      response = original_embedding_create(*args, **kwargs)
    except Exception as exc:
      response = None
      status = "error"
      error = repr(exc)
      raise
    finally:
      end_time_ns = time.time_ns()
      DIRECT.perf.record(
          type="worker_embedding",
          agent=DIRECT.current_agent(),
          step=DIRECT.step,
          status=status,
          error=error,
          start_time_ns=start_time_ns,
          end_time_ns=end_time_ns,
          latency_ms=(end_time_ns - start_time_ns) / 1_000_000,
      )
    return response

  openai.Completion.create = staticmethod(direct_completion_create)
  openai.ChatCompletion.create = staticmethod(direct_chat_create)
  openai.Embedding.create = staticmethod(direct_embedding_create)


def _install_class_perf_hooks():
  from persona.persona import Persona
  from persona.memory_structures.associative_memory import AssociativeMemory
  from reverie import ReverieServer

  if getattr(Persona.move, "_direct_perf_wrapped", False):
    return

  original_move = Persona.move
  original_retrieve = Persona.retrieve
  original_reflect = Persona.reflect
  original_write_headless_environment = ReverieServer.write_headless_environment

  def direct_move(self, maze, personas, curr_tile, curr_time):
    step = DIRECT.current_step()
    DIRECT.set_agent_context(self.name, step)
    start_time_ns = time.time_ns()
    move_started = time.perf_counter()
    status = "ok"
    try:
      output = original_move(self, maze, personas, curr_tile, curr_time)
      return output
    except Exception:
      status = "error"
      raise
    finally:
      end_time_ns = time.time_ns()
      latency_ms = (time.perf_counter() - move_started) * 1000.0
      DIRECT.record_move_window(step, self.name, start_time_ns, end_time_ns, latency_ms)
      DIRECT.perf.record(
          type="direct_agent_move",
          agent=self.name,
          step=step,
          status=status,
          start_time_ns=start_time_ns,
          end_time_ns=end_time_ns,
          latency_ms=latency_ms,
      )
      DIRECT.perf.record(
          type="worker_agent_move_total",
          agent=self.name,
          step=step,
          status=status,
          start_time_ns=start_time_ns,
          end_time_ns=end_time_ns,
          latency_ms=latency_ms,
      )
      DIRECT.clear_agent_context()

  def direct_retrieve(self, perceived):
    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    try:
      return original_retrieve(self, perceived)
    except Exception:
      status = "error"
      raise
    finally:
      end_time_ns = time.time_ns()
      DIRECT.perf.record(
          type="worker_retrieval",
          agent=DIRECT.current_agent(),
          step=DIRECT.step,
          status=status,
          start_time_ns=start_time_ns,
          end_time_ns=end_time_ns,
          latency_ms=(time.perf_counter() - started) * 1000.0,
      )

  def direct_reflect(self):
    start_time_ns = time.time_ns()
    started = time.perf_counter()
    status = "ok"
    try:
      return original_reflect(self)
    except Exception:
      status = "error"
      raise
    finally:
      end_time_ns = time.time_ns()
      DIRECT.perf.record(
          type="worker_reflection",
          agent=DIRECT.current_agent(),
          step=DIRECT.step,
          status=status,
          start_time_ns=start_time_ns,
          end_time_ns=end_time_ns,
          latency_ms=(time.perf_counter() - started) * 1000.0,
      )

  def make_direct_add(method_name, original_method):
    def direct_add(self, *args, **kwargs):
      start_time_ns = time.time_ns()
      started = time.perf_counter()
      status = "ok"
      try:
        return original_method(self, *args, **kwargs)
      except Exception:
        status = "error"
        raise
      finally:
        end_time_ns = time.time_ns()
        DIRECT.perf.record(
            type="worker_memory",
            agent=DIRECT.current_agent(),
            step=DIRECT.step,
            memory_kind=method_name.replace("add_", ""),
            status=status,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    direct_add._direct_perf_wrapped = True
    return direct_add

  def direct_write_headless_environment(self, movements):
    step = getattr(self, "step", None)
    result = original_write_headless_environment(self, movements)
    DIRECT.flush_step(step)
    return result

  direct_move._direct_perf_wrapped = True
  direct_retrieve._direct_perf_wrapped = True
  direct_reflect._direct_perf_wrapped = True
  direct_write_headless_environment._direct_perf_wrapped = True

  Persona.move = direct_move
  Persona.retrieve = direct_retrieve
  Persona.reflect = direct_reflect
  ReverieServer.write_headless_environment = direct_write_headless_environment

  for method_name in ("add_event", "add_thought", "add_chat"):
    original_method = getattr(AssociativeMemory, method_name, None)
    if callable(original_method) and not getattr(original_method, "_direct_perf_wrapped", False):
      setattr(
          AssociativeMemory,
          method_name,
          make_direct_add(method_name, original_method),
      )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--fork", help="Simulation folder to fork from.")
  parser.add_argument("--sim", help="New simulation folder name.")
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument(
      "--run-dir",
      help="Defaults to <this script dir>/direct_run_perf_runs/<sim>.",
  )
  parser.add_argument("--perf", help="Defaults to <run-dir>/perf.jsonl.")
  parser.add_argument("--no-save", action="store_true")
  args = parser.parse_args()

  fork = args.fork or input("Enter the name of the forked simulation: ").strip()
  sim = args.sim or input("Enter the name of the new simulation: ").strip()
  run_dir = args.run_dir or os.path.join(RUN_ROOT, sim)
  perf_path = args.perf or os.path.join(run_dir, "perf.jsonl")
  os.makedirs(run_dir, exist_ok=True)

  with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as outfile:
    json.dump(
        {
            "mode": "direct_headless_perf",
            "fork": fork,
            "sim": sim,
            "steps": args.steps,
            "run_dir": os.path.abspath(run_dir),
            "perf": os.path.abspath(perf_path),
        },
        outfile,
        indent=2,
    )

  global DIRECT
  DIRECT = DirectPerfController(perf_path)

  sglang_openai_patch.HEADLESS_ENVIRONMENT = False
  sglang_openai_patch._install_logging()
  sglang_openai_patch._install_default_utils_if_missing()
  sglang_openai_patch._install_fast_fork_copy()
  sglang_openai_patch._install_reverie_server_init_hook()

  _install_prompt_perf_hooks()

  from reverie import ReverieServer

  _install_openai_perf_hooks()
  _install_class_perf_hooks()

  start_time_ns = time.time_ns()
  status = "ok"
  error = None
  try:
    print(f"[direct perf] fork={fork} sim={sim} steps={args.steps}")
    print(f"[direct perf] run_dir={os.path.abspath(run_dir)}")
    server = ReverieServer(fork, sim)
    DIRECT.set_server(server)
    server.start_server(args.steps)
    if not args.no_save:
      server.save()
    print(f"[direct perf] complete: {sim}")
    print(f"[direct perf] perf: {os.path.abspath(perf_path)}")
  except Exception:
    status = "error"
    error = traceback.format_exc()
    DIRECT.perf.record(
        type="direct_run_error",
        status="error",
        error=error,
    )
    raise
  finally:
    end_time_ns = time.time_ns()
    llm_ms = DIRECT.perf.latency_by_type.get("worker_llm", 0.0)
    embedding_ms = DIRECT.perf.latency_by_type.get("worker_embedding", 0.0)
    wall_ms = (end_time_ns - start_time_ns) / 1_000_000
    DIRECT.perf.record(
        type="direct_run_total",
        status=status,
        error=error,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        latency_ms=wall_ms,
        llm_total_ms=llm_ms,
        embedding_total_ms=embedding_ms,
        non_llm_total_ms=max(0.0, wall_ms - llm_ms),
        overhead_excluding_llm_embedding_ms=max(0.0, wall_ms - llm_ms - embedding_ms),
        event_counts=DIRECT.perf.counts,
        status_counts=DIRECT.perf.status_counts,
    )
    DIRECT.close()


if __name__ == "__main__":
  main()
