"""
Monkey patch the legacy openai==0.27 client used by this project so it talks to
local SGLang OpenAI-compatible servers.

Usage, from reverie/backend_server:
  python -c "import sglang_openai_patch; import reverie"

Or create a tiny local runner that imports this file before importing/running
the simulation code.
"""
import os
import re
import runpy
import shutil
import sys
import types
import traceback
import builtins
import json
import threading

import openai
import requests


# Fill this if your SGLang servers were started with --api-key.
SGLANG_API_KEY = "dummy"

# Set True to speed up forking from saved daytime simulations by not copying
# thousands of old movement/*.json files into the new simulation.
#FAST_FORK_SKIP_OLD_MOVEMENT = False
FAST_FORK_SKIP_OLD_MOVEMENT = True

# Set True to run the backend without the browser/Django visual loop.
# The patch will synthesize environment/<step+1>.json from movement/<step>.json.
#HEADLESS_ENVIRONMENT = False
HEADLESS_ENVIRONMENT = True

SGLANG_API_BASE = os.environ.get("SGLANG_API_BASE", "http://127.0.0.1:1919/v1")
SGLANG_MODEL = os.environ.get("SGLANG_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")

SGLANG_EMBEDDING_API_BASE = os.environ.get(
    "SGLANG_EMBEDDING_API_BASE",
    "http://127.0.0.1:1920/v1",
)
SGLANG_EMBEDDING_MODEL = os.environ.get(
    "SGLANG_EMBEDDING_MODEL",
    "Qwen/Qwen3-Embedding-0.6B",
)
# Saved simulations in this repo usually contain OpenAI ada-002 embeddings
# with 1536 dimensions. Qwen3-Embedding-0.6B returns 1024 dimensions, so we
# pad or truncate new embeddings to this size for compatibility with old memory.
EMBEDDING_DIMENSION = 1536
SGLANG_PATCH_LOG = os.environ.get("SGLANG_PATCH_LOG", "sglang_patch.log")
SGLANG_EMBEDDING_API_KEY = SGLANG_API_KEY
_REQUEST_CONTEXT = threading.local()


def _env_flag(name):
  return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def clear_last_llm_payload():
  _REQUEST_CONTEXT.last_llm_payload = None


def get_last_llm_payload():
  return getattr(_REQUEST_CONTEXT, "last_llm_payload", None)


def _compact_llm_payload(api, payload):
  return {
      "api": api,
      "model": payload.get("model"),
      "temperature": payload.get("temperature"),
      "top_p": payload.get("top_p"),
      "seed": payload.get("seed"),
      "max_tokens": payload.get("max_tokens"),
      "stop": payload.get("stop"),
      "stream": payload.get("stream"),
  }


def _force_temperature(payload):
  value = os.environ.get("SGLANG_FORCE_TEMPERATURE")
  if value is None or value == "":
    return
  try:
    payload["temperature"] = float(value)
  except ValueError:
    payload["temperature"] = 0


def _force_sampling_params(payload):
  _force_temperature(payload)
  top_p = os.environ.get("SGLANG_FORCE_TOP_P")
  if top_p not in (None, ""):
    try:
      payload["top_p"] = float(top_p)
    except ValueError:
      payload["top_p"] = 1
  seed = os.environ.get("SGLANG_REQUEST_SEED")
  if seed not in (None, ""):
    try:
      payload["seed"] = int(seed)
    except ValueError:
      payload["seed"] = seed


openai.api_key = SGLANG_API_KEY or "EMPTY"
openai.api_base = SGLANG_API_BASE


class _Tee:
  def __init__(self, *streams):
    self.streams = streams

  def write(self, data):
    for stream in self.streams:
      stream.write(data)
      stream.flush()

  def flush(self):
    for stream in self.streams:
      stream.flush()


def _log(message):
  with open(SGLANG_PATCH_LOG, "a", encoding="utf-8") as log_file:
    log_file.write(message.rstrip() + "\n")


def _install_logging():
  log_file = open(SGLANG_PATCH_LOG, "w", encoding="utf-8", buffering=1)
  log_file.write("\n=== sglang_openai_patch session start ===\n")
  sys.stdout = _Tee(sys.stdout, log_file)
  sys.stderr = _Tee(sys.stderr, log_file)


def _install_default_utils_if_missing():
  if "utils" in sys.modules:
    return

  try:
    __import__("utils")
    return
  except ModuleNotFoundError:
    pass

  utils = types.ModuleType("utils")
  utils.openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
  utils.key_owner = os.environ.get("KEY_OWNER", "local")

  utils.maze_assets_loc = os.environ.get(
      "MAZE_ASSETS_LOC",
      "../../environment/frontend_server/static_dirs/assets",
  )
  utils.env_matrix = f"{utils.maze_assets_loc}/the_ville/matrix"
  utils.env_visuals = f"{utils.maze_assets_loc}/the_ville/visuals"

  utils.fs_storage = os.environ.get(
      "FS_STORAGE",
      "../../environment/frontend_server/storage",
  )
  utils.fs_temp_storage = os.environ.get(
      "FS_TEMP_STORAGE",
      "../../environment/frontend_server/temp_storage",
  )

  utils.collision_block_id = os.environ.get("COLLISION_BLOCK_ID", "32125")
  #utils.debug = os.environ.get("DEBUG", "True").lower() == "true"
  utils.debug = os.environ.get("DEBUG", "False").lower() == "true"


  sys.modules["utils"] = utils


_embedding_create = openai.Embedding.create


class _OpenAICompatObject(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as exc:
      raise AttributeError(name) from exc


def _to_openai_compat(value):
  if isinstance(value, dict):
    return _OpenAICompatObject(
        {key: _to_openai_compat(val) for key, val in value.items()}
    )
  if isinstance(value, list):
    return [_to_openai_compat(item) for item in value]
  return value


def _post_sglang(path, payload, timeout=300):
  response = requests.post(
      f"{SGLANG_API_BASE.rstrip('/')}/{path.lstrip('/')}",
      headers={
          "Authorization": f"Bearer {SGLANG_API_KEY or 'EMPTY'}",
          "Content-Type": "application/json",
      },
      json=payload,
      timeout=timeout,
  )

  if response.status_code >= 400:
    _log(f"SGLang request failed: {response.status_code} {response.text}")
    print(f"SGLang request failed: {response.status_code} {response.text}")
    response.raise_for_status()

  return response.json()


def _sanitize_completion_text(text, prompt=None):
  for marker in ("\n---", "\nName:", "\nThis breakdown covers"):
    if marker in text:
      text = text.split(marker, 1)[0]
  text = text.strip()

  if prompt:
    match = re.search(r"MUST pick one of \{([^}]+)\}", prompt)
    if match:
      options = [item.strip() for item in match.group(1).split(",")]
      had_closing_brace = "}" in text or "Answer: {" in prompt
      text_without_brace = text.split("}", 1)[0].strip()
      if text_without_brace not in options:
        lowered = {option.lower(): option for option in options}
        text_without_brace = lowered.get(text_without_brace.lower(), options[0])
      text = text_without_brace + ("}" if had_closing_brace else "")

  return text.rstrip()


def _patched_chat_create(*args, **kwargs):
  payload = dict(kwargs)
  payload["model"] = SGLANG_MODEL
  _force_sampling_params(payload)
  _REQUEST_CONTEXT.last_llm_payload = _compact_llm_payload("chat", payload)
  return _to_openai_compat(_post_sglang("chat/completions", payload))


def _patched_completion_create(*args, **kwargs):
  prompt = kwargs.get("prompt")
  if prompt is None and args:
    prompt = args[0]

  payload = {
      "model": SGLANG_MODEL,
      "prompt": prompt or "",
      "temperature": kwargs.get("temperature", 0),
      "max_tokens": kwargs.get("max_tokens", 256),
      "top_p": kwargs.get("top_p", 1),
      "stream": False,
  }

  if kwargs.get("stop") is not None:
    payload["stop"] = kwargs["stop"]
  _force_sampling_params(payload)
  _REQUEST_CONTEXT.last_llm_payload = _compact_llm_payload("completion", payload)

  max_tokens_cap = os.environ.get("SGLANG_MAX_TOKENS_CAP")
  if max_tokens_cap:
    payload["max_tokens"] = min(payload["max_tokens"], int(max_tokens_cap))
    _REQUEST_CONTEXT.last_llm_payload = _compact_llm_payload("completion", payload)

  data = _post_sglang("completions", payload)
  if data.get("choices"):
    for choice in data["choices"]:
      if "text" in choice:
        choice["text"] = _sanitize_completion_text(choice["text"], prompt)
  return _to_openai_compat(data)


def _patched_embedding_create(*args, **kwargs):
  input_text = kwargs.get("input")
  if input_text is None and args:
    input_text = args[0]

  payload = {
      "model": SGLANG_EMBEDDING_MODEL,
      "input": input_text,
  }
  response = requests.post(
      f"{SGLANG_EMBEDDING_API_BASE.rstrip('/')}/embeddings",
      headers={"Authorization": f"Bearer {SGLANG_EMBEDDING_API_KEY or 'EMPTY'}"},
      json=payload,
      timeout=120,
  )

  if response.status_code >= 400:
    error_message = (
        "SGLang embedding request failed: "
        f"{response.status_code} {response.text}"
    )
    _log(error_message)
    raise RuntimeError(error_message)

  data = response.json()
  if EMBEDDING_DIMENSION:
    for item in data.get("data", []):
      embedding = item.get("embedding")
      if embedding is None:
        continue
      if len(embedding) < EMBEDDING_DIMENSION:
        item["embedding"] = embedding + [0.0] * (EMBEDDING_DIMENSION - len(embedding))
      elif len(embedding) > EMBEDDING_DIMENSION:
        item["embedding"] = embedding[:EMBEDDING_DIMENSION]
  return data


openai.ChatCompletion.create = staticmethod(_patched_chat_create)
openai.Completion.create = staticmethod(_patched_completion_create)
openai.Embedding.create = staticmethod(_patched_embedding_create)


def _install_reverie_server_init_hook():
  original_build_class = builtins.__build_class__

  def patched_build_class(func, name, *args, **kwargs):
    cls = original_build_class(func, name, *args, **kwargs)
    if name == "ReverieServer":
      original_init = cls.__init__

      def patched_init(self, fork_sim_code, sim_code):
        original_init(self, fork_sim_code, sim_code)
        fs_storage = sys.modules["utils"].fs_storage
        os.makedirs(os.path.join(fs_storage, sim_code, "movement"), exist_ok=True)

      cls.__init__ = patched_init
    return cls

  builtins.__build_class__ = patched_build_class


def _install_headless_environment():
  if not HEADLESS_ENVIRONMENT:
    return

  original_open = builtins.open

  class MovementWriter:
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
        movement = json.loads(data)
        step = int(os.path.splitext(os.path.basename(self.path))[0])
        sim_folder = os.path.dirname(os.path.dirname(self.path))
        env = {}
        for persona_name, payload in movement.get("persona", {}).items():
          x, y = payload["movement"]
          env[persona_name] = {"maze": "the_ville", "x": x, "y": y}
        next_env_file = os.path.join(sim_folder, "environment", f"{step + 1}.json")
        with original_open(next_env_file, "w") as outfile:
          outfile.write(json.dumps(env, indent=2))
      except Exception:
        _log(traceback.format_exc())
        raise
      return result

  def patched_open(file, mode="r", *args, **kwargs):
    wrapped = original_open(file, mode, *args, **kwargs)
    if (
        "w" in mode
        and isinstance(file, str)
        and f"{os.sep}movement{os.sep}" in file
        and file.endswith(".json")
    ):
      return MovementWriter(wrapped, file)
    return wrapped

  builtins.open = patched_open


def _install_fast_fork_copy():
  if not FAST_FORK_SKIP_OLD_MOVEMENT:
    return

  import global_methods

  def copy_without_old_movements(src, dst):
    if os.path.exists(dst):
      raise FileExistsError(dst)

    os.makedirs(dst)
    for name in os.listdir(src):
      src_path = os.path.join(src, name)
      dst_path = os.path.join(dst, name)

      if name == "movement":
        os.makedirs(dst_path, exist_ok=True)
      elif os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path)
      else:
        shutil.copy2(src_path, dst_path)

  global_methods.copyanything = copy_without_old_movements


def main():
  _install_logging()
  _install_default_utils_if_missing()
  _install_fast_fork_copy()
  _install_reverie_server_init_hook()
  _install_headless_environment()
  try:
    runpy.run_path("reverie.py", run_name="__main__")
  except Exception:
    _log(traceback.format_exc())
    raise


if __name__ == "__main__":
  main()
