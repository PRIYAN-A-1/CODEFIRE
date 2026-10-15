#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_environment.py
printf 'Run source .venv/bin/activate, then python scripts/smoke_test.py\n'
