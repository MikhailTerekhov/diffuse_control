"""Entry point for OpenEvolve with our Inspect-backed LLM provider registered.

Replaces the vendored `thirdparty/evolve/openevolve-run.py` that the research repo shelled
out to. Same CLI as upstream's openevolve-run, so run.py passes the same arguments:

    python -m max_weak.openevolve_main <initial_program> <evaluator> \
        --config <yaml> --output <dir> --iterations <n> [--checkpoint <dir>]
"""

import sys

from max_weak.inspect_llm import register


def main() -> int:
    register()
    from openevolve.cli import main as openevolve_main
    return openevolve_main()


if __name__ == "__main__":
    sys.exit(main())
