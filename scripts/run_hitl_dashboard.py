#!/usr/bin/env python
"""Launch the HITL supervisor dashboard via Streamlit.

Usage:
    uv run python scripts/run_hitl_dashboard.py --artifact-dir artifacts/hitl_demo/supervisor

This wraps ``streamlit run`` so the ``python/`` sources are importable and the artifact
directory is passed through to the app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO / "python"
DASHBOARD = PYTHON_DIR / "learning" / "dashboard.py"

DEFAULT_ARTIFACT_DIR = "artifacts/hitl_demo/supervisor"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the PharmaSim HITL dashboard.")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.stderr.write(
            "Streamlit is not installed. Install the dashboard extra:\n"
            "    uv sync --extra dashboard\n"
        )
        raise SystemExit(1)

    sys.argv = [
        "streamlit",
        "run",
        str(DASHBOARD),
        "--server.port",
        str(args.port),
        "--",
        args.artifact_dir,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
