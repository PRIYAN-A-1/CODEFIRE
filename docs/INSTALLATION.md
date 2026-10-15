# Installation notes

The tested dependency versions are pinned in requirements.txt and package-lock.json. Python 3.12/Linux CPU was executed. Installation for CUDA/native Windows is supplied but not run. Node 22+ is supported by the Vite version; Node 24 was used in verification. GPU package selection depends on your driver/runtime: no CUDA wheel index is forced by setup scripts.

The setup scripts create a venv, install pinned Python dependencies and print measured hardware. Smoke tests are an explicit second step to avoid silently launching training during installation. Use `python scripts/smoke_test.py`. React dashboard build: `cd frontend`, `npm ci`, `npm run build`. Built files are included for inspection; to serve an installed frontend with backend proxying outside Vite, configure a local reverse proxy yourself.

The source ZIP has no downloaded training corpus or external weights. Included artifacts are tiny original fixture BPE and actual smoke checkpoints. Dependency installation and full model training need additional disk beyond the archive. Paths in the saved hardware report describe this build environment, not your laptop.
