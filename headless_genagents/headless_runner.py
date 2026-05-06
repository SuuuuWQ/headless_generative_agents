"""
Run the original Generative Agents backend loop without the visual frontend.

Example:
  python headless_runner.py --fork test3 --sim headless_test --steps 20
"""
import argparse

import sglang_openai_patch

sglang_openai_patch.HEADLESS_ENVIRONMENT = False
sglang_openai_patch._install_logging()
sglang_openai_patch._install_default_utils_if_missing()
sglang_openai_patch._install_fast_fork_copy()
sglang_openai_patch._install_reverie_server_init_hook()

from reverie import ReverieServer


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--fork", help="Simulation folder to fork from.")
  parser.add_argument("--sim", help="New simulation folder name.")
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument("--no-save", action="store_true")
  args = parser.parse_args()

  fork = args.fork or input("Enter the name of the forked simulation: ").strip()
  sim = args.sim or input("Enter the name of the new simulation: ").strip()

  server = ReverieServer(fork, sim)
  server.start_server(args.steps)
  if not args.no_save:
    server.save()

  print(f"Headless run complete: {sim}, steps={args.steps}")


if __name__ == "__main__":
  main()
