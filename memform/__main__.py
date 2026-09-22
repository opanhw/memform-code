"""MemForm CLI: ``python -m memform <command> [options]``."""
from __future__ import annotations

import sys

from memform.common import load_env

COMMANDS = {
    "generate": "memform.generate",
    "compile": "memform.compile",
    "build-freeform": "memform.freeform.build",
    "validate-mcq": "memform.audit.mcq",
    "validate-freeform": "memform.audit.freeform",
    "run-langchain": "memform.eval.core",
    "run-llamaindex": "memform.eval.llamaindex",
    "run-defense": "memform.eval.defense",
}

HELP = """\
MemForm - deterministic persistent-memory poisoning benchmark

Usage:
  python -m memform <command> [options]

Commands:
  generate           LLM-assisted family specification
  compile            Zero-API compiler (family specs -> P1-P5 cases)
  build-freeform     Deterministic MCQ -> free-form conversion
  validate-mcq       Blind MCQ construction audit
  validate-freeform  Blind free-form construction audit
  run-langchain      Evaluate with the LangChain stack
  run-llamaindex     Evaluate with the LlamaIndex stack
  run-defense        Memory defense evaluation (MCQ; write-time and use-time)

Equivalent root scripts (generate_family_specs.py, compile_benchmark.py, ...)
call the same entry points.
"""


def main() -> None:
    load_env()
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(HELP, end="")
        raise SystemExit(0 if len(sys.argv) > 1 else 2)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command {cmd!r}\n")
        print(HELP, end="")
        raise SystemExit(2)
    import importlib

    sys.argv = [f"python -m memform {cmd}"] + sys.argv[2:]
    module = importlib.import_module(COMMANDS[cmd])
    module.main()


if __name__ == "__main__":
    main()
