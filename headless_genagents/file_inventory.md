# Headless GA File Inventory

This note records what was copied from the original Generative Agents codebase
into `headless_genagents`, what was modified, and what external files still
matter for a headless run.

## Copied From `reverie/backend_server`

| Original file or directory | Copied to `headless_genagents` | Modified in copy |
|---|---:|---:|
| `global_methods.py` | Yes | No |
| `maze.py` | Yes | No |
| `path_finder.py` | Yes | No |
| `reverie.py` | Yes | Yes |
| `sglang_openai_patch.py` | Yes | No |
| `test.py` | No | - |
| `trace_reverie_runner.py` | No | - |
| `persona/persona.py` | Yes | No |
| `persona/cognitive_modules/converse.py` | Yes | No |
| `persona/cognitive_modules/execute.py` | Yes | No |
| `persona/cognitive_modules/perceive.py` | Yes | No |
| `persona/cognitive_modules/plan.py` | Yes | No |
| `persona/cognitive_modules/reflect.py` | Yes | No |
| `persona/cognitive_modules/retrieve.py` | Yes | No |
| `persona/memory_structures/associative_memory.py` | Yes | No |
| `persona/memory_structures/scratch.py` | Yes | No |
| `persona/memory_structures/spatial_memory.py` | Yes | No |
| `persona/prompt_template/*.py` | Yes | No |
| `persona/prompt_template/**/*.txt` | Yes | No |

The full `persona/` tree was copied unchanged, including cognitive modules,
memory structures, prompt template Python files, and prompt text files.

## Modified Copied Files

| File | Changes |
|---|---|
| `headless_genagents/reverie.py` | Removed the Selenium webdriver import; added `write_headless_environment()`; after each `movement/<step>.json` write, the backend now writes `environment/<step + 1>.json` itself; changed `rs.start_server(...)` to `self.start_server(...)`; ensures temp and movement directories exist before writing files. |

## New Files

| New file | Purpose |
|---|---|
| `headless_genagents/utils.py` | Local configuration for the copied headless runtime. By default, it points to the original repository's `environment/frontend_server/storage` and `static_dirs/assets`. |
| `headless_genagents/headless_runner.py` | Command-line entry point for running the copied GA backend without the visual frontend. |
| `headless_genagents/headless_trace_runner.py` | Command-line entry point for running the copied headless GA backend while recording replay trace events. It was copied from `reverie/backend_server/trace_reverie_runner.py` and adapted to call `ReverieServer` directly instead of launching the original interactive `reverie.py`. |
| `headless_genagents/headless_replay_runner.py` | First trace-guided replay runner. It runs the copied headless GA control flow, executes LLM/embedding/random nodes for timing, returns canonical LLM/random outputs, canonicalizes retrieval by recorded node ids, and validates movement against the trace. |
| `headless_genagents/verify_headless_replay.py` | Automatic verifier for trace-guided replay output. It checks trace integrity, movement parity, meta consistency, perf log counts/statuses, chat coverage, memory kinds, retrieval coverage, and prompt-template coverage. By default it looks for perf logs named after the replay simulation. |
| `headless_genagents/README.md` | Short usage note for the headless copy. |
| `headless_genagents/file_inventory.md` | This inventory. |

## Original Backend Files Not Copied

| Original path | Reason |
|---|---|
| `reverie/backend_server/test.py` | Not part of the main GA runtime. |
| `reverie/backend_server/trace_reverie_runner.py` | Trace recorder patch, not part of the minimal headless GA body. It can be ported later as a replay/trace hook. |
| `reverie/backend_server/traces/*.jsonl` | Generated trace data, not code. |
| `reverie/backend_server/sglang_patch.log` | Runtime log, not code. |
| `reverie/backend_server/__pycache__/` | Python cache. |

## Files Outside `reverie/backend_server`

### Repository Root

| Path | Purpose | Needed for headless GA |
|---|---|---:|
| `README.md` | Original project documentation. | No |
| `requirements.txt` | Original Python dependency reference. | Useful |
| `LICENSE` | License. | No |
| `cover.png` | README image. | No |
| `.gitignore`, `.gitattributes` | Git config. | No |

### `reverie/` Top Level

| Path | Purpose | Needed for headless GA |
|---|---|---:|
| `reverie/backend_server/` | Original GA backend: agents, memory, planning, prompts, maze logic. | Yes |
| `reverie/compress_sim_storage.py` | Utility for compressing simulation storage. | No for now |
| `reverie/global_methods.py` | Utility-function copy similar to the backend one. | No for now |

### Frontend/Django Code

| Path | Purpose | Needed for headless GA |
|---|---|---:|
| `environment/frontend_server/manage.py` | Django frontend entry point. | No |
| `environment/frontend_server/db.sqlite3` | Django database. | No |
| `environment/frontend_server/requirements.txt` | Frontend/Django dependencies. | No |
| `environment/frontend_server/Procfile`, `runtime.txt` | Deployment config. | No |
| `environment/frontend_server/global_methods.py` | Frontend-side utility functions. | No |
| `environment/frontend_server/frontend_server/` | Django project config, urls, settings, wsgi. | No |
| `environment/frontend_server/translator/` | Django app for frontend pages/backend interaction. | No |
| `environment/frontend_server/templates/` | HTML templates. | No |
| `environment/frontend_server/static_dirs/css/` | Frontend CSS. | No |
| `environment/frontend_server/static_dirs/img/` | Frontend image assets. | No |

### Data And Assets

| Path | Purpose | Needed for headless GA |
|---|---|---:|
| `environment/frontend_server/storage/` | Simulation saves: `reverie/meta.json`, persona memory, environment, movement. | Yes |
| `environment/frontend_server/static_dirs/assets/the_ville/matrix/` | Map matrix/tile metadata: collision, sector, arena, object data. | Yes |
| `environment/frontend_server/static_dirs/assets/the_ville/visuals/` | Visual map resources. Mostly for frontend; may be kept with assets for compatibility. | Maybe |
| `environment/frontend_server/static_dirs/assets/characters/` | Character sprites/profile images. | No for headless |
| `environment/frontend_server/temp_storage/` | Original frontend/backend temp communication files. | Weak dependency; copied headless code still writes some files, but does not rely on the visual frontend. |
| `environment/frontend_server/compressed_storage/` | Compressed simulation storage. | No for now |

### Research/Replay Code

| Path | Purpose | Needed for headless GA |
|---|---|---:|
| `replay_genagents/` | Phase-1 replay player, performance recorder, and replay design docs. | Not for ordinary headless GA; useful for replay experiments |
| `headless_genagents/` | Copied headless GA branch. | Yes |
| `reverie/backend_server/traces/` | Trace JSONL files. | Needed for replay, not for free-running headless GA |

## Current Migration Boundary

The copied `headless_genagents` code is isolated from the original backend code,
but it still uses the original repository's large data directories by default:

```text
environment/frontend_server/storage/
environment/frontend_server/static_dirs/assets/
```

This keeps the first headless version lightweight. To make it fully separable
later, copy the selected storage folders and map assets into `headless_genagents`
and update `fs_storage` / `maze_assets_loc` in `headless_genagents/utils.py`.
