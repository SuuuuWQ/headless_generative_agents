"""
Run a trace-driven LLM benchmark with interaction-group scheduling.

This does not run GA simulation logic. It reads LLM calls from a trace, restores
their prompts when possible, then sends only those LLM requests to the configured
OpenAI-compatible backend. Steps run in trace order; groups within a step run in
parallel; agents within a group run serially; each agent's LLM calls run serially.

Example:
  python run_grouped_trace_llm.py direct_trace_perf_n25_10_f

  # Equivalent explicit-path form:
  python run_grouped_trace_llm.py \
    runs/direct_trace_perf_n25_10_f/trace.jsonl \
    runs/direct_trace_perf_n25_10_f/group_index.jsonl
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import traceback
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(SCRIPT_DIR, "runs")
OPENAI = None
SGLANG_PATCH = None


def _now_ns():
  return time.time_ns()


def _latency_ms(start_ns, end_ns):
  return (end_ns - start_ns) / 1_000_000


def _sha256_text(text):
  return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _safe_name(value):
  value = str(value or "").strip()
  return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _read_jsonl(path):
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


def _append_jsonl(path, record, lock):
  with lock:
    with open(path, "a", encoding="utf-8", buffering=1) as outfile:
      outfile.write(json.dumps(record, ensure_ascii=True) + "\n")


def _resolve_path(path, base_dir=SCRIPT_DIR):
  if not path:
    return None
  if os.path.exists(path):
    return os.path.abspath(path)
  candidate = os.path.join(base_dir, path)
  if os.path.exists(candidate):
    return os.path.abspath(candidate)
  normalized = path.replace("\\", "/")
  if normalized.startswith("/mnt/") and len(normalized) > 6:
    drive = normalized[5]
    rest = normalized[7:].replace("/", os.sep)
    candidate = f"{drive.upper()}:{os.sep}{rest}"
    if os.path.exists(candidate):
      return os.path.abspath(candidate)
  return os.path.abspath(path)


def _resolve_prompt_template(path):
  return _resolve_path(path, SCRIPT_DIR)


def _first_existing(paths):
  for path in paths:
    if path and os.path.exists(path):
      return os.path.abspath(path)
  return None


def _infer_trace_and_group_paths(trace_or_sim, group_index=None):
  if group_index:
    return _resolve_path(trace_or_sim), _resolve_path(group_index)

  resolved_input = _resolve_path(trace_or_sim)
  looks_like_file = (
      os.path.exists(resolved_input)
      or trace_or_sim.endswith(".jsonl")
      or os.path.sep in trace_or_sim
      or "/" in trace_or_sim
      or "\\" in trace_or_sim
  )
  if looks_like_file:
    raise SystemExit(
        "When the first argument is a trace path, group_index is required.\n"
        "For simple mode, pass the sim name, for example:\n"
        "  python3 run_grouped_trace_llm.py direct_trace_perf_n25_10_f"
    )

  sim = trace_or_sim
  trace_candidates = [
      os.path.join(RUN_ROOT, sim, "llm_trace.jsonl"),
      os.path.join(RUN_ROOT, sim, "trace.jsonl"),
      os.path.join(SCRIPT_DIR, "traces", f"trace_{sim}.jsonl"),
      os.path.join(SCRIPT_DIR, "traces", f"{sim}.jsonl"),
  ]
  storage_root = os.path.abspath(
      os.path.join(SCRIPT_DIR, "..", "environment", "frontend_server", "storage")
  )
  group_candidates = [
      os.path.join(RUN_ROOT, sim, "group_index.jsonl"),
      os.path.join(storage_root, sim, "group_index.jsonl"),
  ]

  trace_path = _first_existing(trace_candidates)
  group_path = _first_existing(group_candidates)
  missing = []
  if not trace_path:
    missing.append("trace: " + " or ".join(trace_candidates))
  if not group_path:
    missing.append("group_index: " + " or ".join(group_candidates))
  if missing:
    raise SystemExit(
        f"Could not infer files for sim '{sim}':\n  " + "\n  ".join(missing)
    )
  return trace_path, group_path


def _generate_prompt(curr_input, prompt_template):
  if isinstance(curr_input, str):
    curr_input = [curr_input]
  curr_input = [str(item) for item in curr_input]
  with open(prompt_template, "r", encoding="utf-8") as infile:
    prompt = infile.read()
  for index, value in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {index}>!", value)
  marker = "<commentblockmarker>###</commentblockmarker>"
  if marker in prompt:
    prompt = prompt.split(marker, 1)[1]
  return prompt.strip()


def _completion_texts(response):
  if not response:
    return []
  try:
    return [
        choice.get("text", "")
        for choice in response.get("choices", [])
    ]
  except AttributeError:
    return []


def _chat_texts(response):
  if not response:
    return []
  texts = []
  try:
    for choice in response.get("choices", []):
      message = choice.get("message") or {}
      texts.append(message.get("content", ""))
  except AttributeError:
    pass
  return texts


def _clean_request_kwargs(request):
  kwargs = {}
  for key, value in (request or {}).items():
    if value is not None:
      kwargs[key] = value
  kwargs.pop("engine", None)
  return kwargs


def _load_trace(trace_path):
  prompts = {}
  llm_calls = []
  responses = {}
  session = {}
  trace_meta = {}

  for record in _read_jsonl(trace_path):
    record_type = record.get("type")
    if record_type == "trace_session_start":
      session = record
    elif record_type == "simulation_init":
      trace_meta["sim_code"] = record.get("sim_code") or trace_meta.get("sim_code")
      trace_meta["fork_sim_code"] = record.get("fork_sim_code") or trace_meta.get("fork_sim_code")
      trace_meta["start_step"] = record.get("step")
      trace_meta["personas"] = record.get("personas")
    elif record_type == "run_start":
      trace_meta["sim_code"] = record.get("sim_code") or trace_meta.get("sim_code")
      trace_meta["run_start_step"] = record.get("start_step")
      trace_meta["requested_steps"] = record.get("requested_steps")
    elif record_type == "run_end":
      trace_meta["run_end_step"] = record.get("end_step")
    elif record_type == "prompt_built":
      prompt_id = record.get("prompt_id")
      if prompt_id:
        prompts[prompt_id] = record
    elif record_type == "llm_request":
      llm_calls.append(record)
    elif record_type == "llm_response":
      responses[record.get("call_id")] = record

  for call in llm_calls:
    call["trace_response"] = responses.get(call.get("call_id"))
  return session, trace_meta, prompts, llm_calls


def _load_group_index(group_index_path):
  meta = {}
  groups_by_step = {}
  for record in _read_jsonl(group_index_path):
    if record.get("type") == "group_index_meta":
      meta = record
    elif record.get("type") == "group_step":
      groups_by_step[int(record["step"])] = record.get("groups") or []
  return meta, groups_by_step


def _restore_prompt(call, prompts):
  request = call.get("request") or {}
  api = call.get("api")

  if api == "completion" and request.get("prompt"):
    return request["prompt"], "request.prompt", None

  if api == "chat":
    messages = request.get("messages")
    if messages:
      return messages, "request.messages", None

  prompt_record = call.get("prompt_record") or {}
  prompt_id = prompt_record.get("prompt_id")
  prompt_built = prompts.get(prompt_id)
  if not prompt_built:
    return None, "missing_prompt_record", f"missing prompt_built for {prompt_id}"

  if "prompt" in prompt_built:
    prompt = prompt_built["prompt"]
    source = "prompt_built.prompt"
  else:
    template = _resolve_prompt_template(prompt_built.get("prompt_template"))
    prompt = _generate_prompt(prompt_built.get("prompt_input") or [], template)
    source = "reconstructed_from_template"

  expected_hash = prompt_record.get("prompt_sha256") or prompt_built.get("prompt_sha256")
  actual_hash = _sha256_text(prompt)
  warning = None
  if expected_hash and actual_hash != expected_hash:
    warning = f"prompt hash mismatch expected={expected_hash} actual={actual_hash}"

  if api == "chat":
    warning = (
        (warning + "; " if warning else "")
        + "chat messages were not recorded; sending restored base prompt as one user message"
    )
    return [{"role": "user", "content": prompt}], source + ".as_chat_message", warning

  return prompt, source, warning


def _effective_request(call):
  response = call.get("trace_response") or {}
  return response.get("effective_request") or call.get("request") or {}


def _run_llm_call(call, prompts, dry_run=False, record_response_text=False):
  start_ns = _now_ns()
  prompt_or_messages, prompt_source, prompt_warning = _restore_prompt(call, prompts)
  restore_ns = _now_ns()

  status = "ok"
  error = None
  response_texts = []
  effective_payload = None

  try:
    if prompt_or_messages is None:
      raise RuntimeError(prompt_warning or "could not restore prompt")

    if not dry_run:
      SGLANG_PATCH.clear_last_llm_payload()
      api = call.get("api")
      request = _clean_request_kwargs(call.get("request") or {})
      if api == "completion":
        request["prompt"] = prompt_or_messages
        response = OPENAI.Completion.create(**request)
        response_texts = _completion_texts(response)
      elif api == "chat":
        request["messages"] = prompt_or_messages
        response = OPENAI.ChatCompletion.create(**request)
        response_texts = _chat_texts(response)
      else:
        raise RuntimeError(f"unsupported api: {api}")
      effective_payload = SGLANG_PATCH.get_last_llm_payload()
    else:
      effective_payload = _effective_request(call)

  except Exception:
    status = "error"
    error = traceback.format_exc()

  end_ns = _now_ns()
  record = {
      "type": "grouped_llm_call",
      "status": status,
      "error": error,
      "step": call.get("step"),
      "agent": call.get("agent"),
      "api": call.get("api"),
      "call_id": call.get("call_id"),
      "trace_seq": call.get("seq"),
      "prompt_record": call.get("prompt_record"),
      "prompt_source": prompt_source,
      "prompt_warning": prompt_warning,
      "prompt_chars": (
          len(prompt_or_messages)
          if isinstance(prompt_or_messages, str)
          else sum(len(str(item.get("content", ""))) for item in (prompt_or_messages or []))
      ),
      "start_time_ns": start_ns,
      "restore_done_ns": restore_ns,
      "end_time_ns": end_ns,
      "restore_ms": _latency_ms(start_ns, restore_ns),
      "latency_ms": _latency_ms(start_ns, end_ns),
      "effective_request": effective_payload,
      "trace_effective_request": _effective_request(call),
      "trace_response_texts": (call.get("trace_response") or {}).get("canonical_texts"),
  }
  if record_response_text:
    record["response_texts"] = response_texts
  else:
    record["response_text_count"] = len(response_texts)
  return record


def _calls_by_step_agent(llm_calls, start_step=None, end_step=None, limit_calls=None):
  selected = []
  for call in llm_calls:
    step = call.get("step")
    if step is None:
      continue
    if start_step is not None and step < start_step:
      continue
    if end_step is not None and step > end_step:
      continue
    selected.append(call)
    if limit_calls is not None and len(selected) >= limit_calls:
      break

  calls = defaultdict(lambda: defaultdict(list))
  for call in selected:
    calls[int(call["step"])][call.get("agent")].append(call)
  for agent_calls in calls.values():
    for call_list in agent_calls.values():
      call_list.sort(key=lambda item: item.get("seq", 0))
  return calls, selected


def _groups_for_step(step, calls_for_step, groups_by_step, allow_missing_groups=False):
  groups = groups_by_step.get(step)
  agents_with_calls = set(calls_for_step)
  if not groups:
    if not allow_missing_groups:
      raise RuntimeError(
          f"group_index has no group_step for LLM step {step}. "
          "Use the matching group_index or pass --allow-missing-groups."
      )
    return [[agent] for agent in sorted(agents_with_calls)]

  result = []
  covered = set()
  for group in groups:
    active = [agent for agent in group if agent in agents_with_calls]
    if active:
      result.append(active)
      covered.update(active)
  for agent in sorted(agents_with_calls - covered):
    result.append([agent])
  return result


def _run_agent(agent, agent_calls, prompts, output_path, output_lock, args, group_id):
  start_ns = _now_ns()
  call_records = []
  for call in agent_calls:
    record = _run_llm_call(
        call,
        prompts,
        dry_run=args.dry_run,
        record_response_text=args.record_response_text,
    )
    record["group_id"] = group_id
    _append_jsonl(output_path, record, output_lock)
    call_records.append(record)
    if record["status"] == "error" and args.stop_on_error:
      raise RuntimeError(record["error"])
  end_ns = _now_ns()
  return {
      "type": "grouped_agent_round",
      "step": agent_calls[0].get("step") if agent_calls else None,
      "agent": agent,
      "group_id": group_id,
      "llm_calls": len(agent_calls),
      "start_time_ns": start_ns,
      "end_time_ns": end_ns,
      "latency_ms": _latency_ms(start_ns, end_ns),
      "error_count": sum(1 for record in call_records if record["status"] != "ok"),
  }


def _run_group(step, group_id, group_agents, calls_for_step, prompts, output_path, output_lock, args):
  start_ns = _now_ns()
  agent_records = []
  for agent in group_agents:
    agent_calls = calls_for_step.get(agent) or []
    if not agent_calls:
      continue
    record = _run_agent(agent, agent_calls, prompts, output_path, output_lock, args, group_id)
    agent_records.append(record)
    _append_jsonl(output_path, record, output_lock)
  end_ns = _now_ns()
  return {
      "type": "grouped_group_round",
      "step": step,
      "group_id": group_id,
      "agents": group_agents,
      "llm_calls": sum(record["llm_calls"] for record in agent_records),
      "start_time_ns": start_ns,
      "end_time_ns": end_ns,
      "latency_ms": _latency_ms(start_ns, end_ns),
      "error_count": sum(record["error_count"] for record in agent_records),
  }


def _run_step(step, calls_for_step, groups, prompts, output_path, output_lock, args):
  start_ns = _now_ns()
  group_records = []
  workers = args.max_workers or max(1, len(groups))
  with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
    futures = []
    for index, group_agents in enumerate(groups):
      futures.append(
          executor.submit(
              _run_group,
              step,
              index,
              group_agents,
              calls_for_step,
              prompts,
              output_path,
              output_lock,
              args,
          )
      )
    for future in concurrent.futures.as_completed(futures):
      group_record = future.result()
      group_records.append(group_record)
      _append_jsonl(output_path, group_record, output_lock)
  end_ns = _now_ns()
  return {
      "type": "grouped_step_round",
      "step": step,
      "group_count": len(groups),
      "active_agent_count": len(calls_for_step),
      "llm_calls": sum(record["llm_calls"] for record in group_records),
      "start_time_ns": start_ns,
      "end_time_ns": end_ns,
      "latency_ms": _latency_ms(start_ns, end_ns),
      "max_group_ms": max((record["latency_ms"] for record in group_records), default=0.0),
      "sum_group_ms": sum(record["latency_ms"] for record in group_records),
      "error_count": sum(record["error_count"] for record in group_records),
  }


def main():
  global OPENAI
  global SGLANG_PATCH

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "trace_or_sim",
      help=(
          "Either a sim name such as direct_trace_perf_n25_10_f, "
          "or an explicit trace JSONL path."
      ),
  )
  parser.add_argument(
      "group_index",
      nargs="?",
      help="Group index JSONL. Required only when trace_or_sim is an explicit trace path.",
  )
  parser.add_argument("--run-dir", help="Defaults to runs/<sim> next to this script.")
  parser.add_argument("--output", help="Defaults to <run-dir>/grouped_trace_llm_perf.jsonl.")
  parser.add_argument("--plot", action="store_true", help="After the run, write timeline figures.")
  parser.add_argument(
      "--allow-mismatch",
      action="store_true",
      help="Allow trace sim_code and group_index sim to differ.",
  )
  parser.add_argument(
      "--allow-missing-groups",
      action="store_true",
      help="If a selected LLM step has no group_step, fall back to one agent per group.",
  )
  parser.add_argument("--start-step", type=int)
  parser.add_argument("--end-step", type=int)
  parser.add_argument("--limit-calls", type=int)
  parser.add_argument("--max-workers", type=int, help="Max parallel groups per step. Defaults to group count.")
  parser.add_argument("--dry-run", action="store_true", help="Restore and schedule calls without sending requests.")
  parser.add_argument("--record-response-text", action="store_true")
  parser.add_argument("--stop-on-error", action="store_true")
  parser.add_argument("--temperature", help="Set SGLANG_FORCE_TEMPERATURE for this process.")
  parser.add_argument("--top-p", help="Set SGLANG_FORCE_TOP_P for this process.")
  parser.add_argument("--seed", help="Set SGLANG_REQUEST_SEED for this process.")
  parser.add_argument("--model", help="Override sglang_openai_patch.SGLANG_MODEL.")
  parser.add_argument("--api-base", help="Override sglang_openai_patch.SGLANG_API_BASE.")
  args = parser.parse_args()

  if not args.dry_run:
    import openai
    import sglang_openai_patch

    OPENAI = openai
    SGLANG_PATCH = sglang_openai_patch
  else:
    OPENAI = None
    SGLANG_PATCH = None

  if args.temperature is not None:
    os.environ["SGLANG_FORCE_TEMPERATURE"] = str(args.temperature)
  if args.top_p is not None:
    os.environ["SGLANG_FORCE_TOP_P"] = str(args.top_p)
  if args.seed is not None:
    os.environ["SGLANG_REQUEST_SEED"] = str(args.seed)
  if args.model:
    if SGLANG_PATCH is not None:
      SGLANG_PATCH.SGLANG_MODEL = args.model
  if args.api_base:
    if SGLANG_PATCH is not None:
      SGLANG_PATCH.SGLANG_API_BASE = args.api_base
      OPENAI.api_base = args.api_base

  trace_path, group_index_path = _infer_trace_and_group_paths(
      args.trace_or_sim,
      args.group_index,
  )
  group_meta, groups_by_step = _load_group_index(group_index_path)
  session, trace_meta, prompts, llm_calls = _load_trace(trace_path)
  calls_by_step, selected_calls = _calls_by_step_agent(
      llm_calls,
      start_step=args.start_step,
      end_step=args.end_step,
      limit_calls=args.limit_calls,
  )

  trace_sim = trace_meta.get("sim_code")
  group_sim = group_meta.get("sim")
  if trace_sim and group_sim and trace_sim != group_sim and not args.allow_mismatch:
    raise SystemExit(
        "Trace and group_index do not match:\n"
        f"  trace sim_code: {trace_sim}\n"
        f"  group_index sim: {group_sim}\n"
        "Use the matching group_index, or pass --allow-mismatch if this is intentional."
    )

  missing_group_steps = [
      step for step in sorted(calls_by_step)
      if step not in groups_by_step
  ]
  if missing_group_steps and not args.allow_missing_groups:
    preview = ", ".join(str(step) for step in missing_group_steps[:10])
    if len(missing_group_steps) > 10:
      preview += ", ..."
    raise SystemExit(
        "group_index is missing group_step records for selected LLM steps:\n"
        f"  {preview}\n"
        "Use the matching group_index, or pass --allow-missing-groups to fall back to one agent per group."
    )

  run_name = _safe_name(trace_sim) or _safe_name(group_sim) or _safe_name(
      os.path.splitext(os.path.basename(trace_path))[0]
  )
  run_dir = args.run_dir or os.path.join(RUN_ROOT, run_name)
  output_path = args.output or os.path.join(run_dir, "grouped_trace_llm_perf.jsonl")
  os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
  if os.path.exists(output_path):
    os.remove(output_path)
  output_lock = threading.Lock()

  start_ns = _now_ns()
  start_record = {
      "type": "grouped_llm_benchmark_start",
      "trace": trace_path,
      "group_index": group_index_path,
      "output": output_path,
      "run_dir": os.path.abspath(run_dir),
      "dry_run": args.dry_run,
      "selected_llm_calls": len(selected_calls),
      "selected_steps": sorted(calls_by_step),
      "trace_session": {
          "sglang_model": session.get("sglang_model"),
          "sglang_api_base": session.get("sglang_api_base"),
          "record_full_prompt": session.get("record_full_prompt"),
      },
      "trace_meta": trace_meta,
      "runtime_model": (
          getattr(SGLANG_PATCH, "SGLANG_MODEL", None)
          if SGLANG_PATCH is not None
          else args.model or os.environ.get("SGLANG_MODEL")
      ),
      "runtime_api_base": (
          getattr(SGLANG_PATCH, "SGLANG_API_BASE", None)
          if SGLANG_PATCH is not None
          else args.api_base or os.environ.get("SGLANG_API_BASE")
      ),
      "runtime_temperature": os.environ.get("SGLANG_FORCE_TEMPERATURE"),
      "runtime_top_p": os.environ.get("SGLANG_FORCE_TOP_P"),
      "runtime_seed": os.environ.get("SGLANG_REQUEST_SEED"),
      "group_meta": {
          "sim": group_meta.get("sim"),
          "maze": group_meta.get("maze"),
          "start_step": group_meta.get("start_step"),
          "end_step": group_meta.get("end_step"),
      },
      "allow_mismatch": args.allow_mismatch,
      "allow_missing_groups": args.allow_missing_groups,
  }
  _append_jsonl(output_path, start_record, output_lock)

  total_errors = 0
  step_records = []
  for step in sorted(calls_by_step):
    calls_for_step = calls_by_step[step]
    groups = _groups_for_step(
        step,
        calls_for_step,
        groups_by_step,
        allow_missing_groups=args.allow_missing_groups,
    )
    print(
        f"[grouped llm] step={step} groups={len(groups)} "
        f"agents={len(calls_for_step)} calls={sum(len(v) for v in calls_for_step.values())}"
    )
    step_record = _run_step(
        step,
        calls_for_step,
        groups,
        prompts,
        output_path,
        output_lock,
        args,
    )
    step_records.append(step_record)
    total_errors += step_record["error_count"]
    _append_jsonl(output_path, step_record, output_lock)
    if total_errors and args.stop_on_error:
      break

  end_ns = _now_ns()
  end_record = {
      "type": "grouped_llm_benchmark_end",
      "status": "error" if total_errors else "ok",
      "error_count": total_errors,
      "steps": len(step_records),
      "llm_calls": sum(record["llm_calls"] for record in step_records),
      "start_time_ns": start_ns,
      "end_time_ns": end_ns,
      "latency_ms": _latency_ms(start_ns, end_ns),
      "sum_step_ms": sum(record["latency_ms"] for record in step_records),
  }
  _append_jsonl(output_path, end_record, output_lock)
  print(f"[grouped llm] wrote {output_path}")
  print(
      f"[grouped llm] status={end_record['status']} "
      f"steps={end_record['steps']} calls={end_record['llm_calls']} "
      f"wall_ms={end_record['latency_ms']:.2f}"
  )
  if args.plot:
    import plot_grouped_trace_llm

    figure_dir = os.path.join(os.path.dirname(output_path), "figures")
    plot_grouped_trace_llm.write_figures(output_path, figure_dir)
    print(f"[grouped llm] figures: {figure_dir}")


if __name__ == "__main__":
  main()
