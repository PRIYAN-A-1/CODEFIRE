# API and dashboard

Run `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` from the project root. Dev UI proxies /api to that local backend. Set `PRIYAN_API_KEY` as an environment variable to require Bearer authentication; enter the key in dashboard Settings (held in memory only). Remote hosting is not turnkey: loopback Host checks, authentication, a trusted reverse proxy, TLS and bounded requests/connections need deployment configuration.

Endpoints: GET /health, /api/system/hardware, /api/system/storage, /api/models, /api/models/{id}, /api/datasets, /api/training/status, /api/checkpoints, /api/evaluation/results; POST /api/datasets/validate, /api/training/start, /api/training/resume, /api/training/stop, /api/generate, /api/generate/stream, /api/evaluation/run. See /docs for schemas.

Generation uses a lock. Training and evaluation use a single managed subprocess slot; they cannot start while inference owns the lock, and inference refuses while a job is running. Configure one API worker only: process/lock state is in memory. Restarting the API while its child runs loses job tracking; stop jobs before restart. Production process supervision is not implemented. Training presets are allowlisted rather than accepting arbitrary shell arguments or filesystem paths from a client. Check logs/job.log for startup failures; status exposes returncode.

Start fresh refuses an existing latest pointer; resume uses each preset's own checkpoint directory. Main preset corpus paths are separate from the fixture. Set PRIYAN_CHECKPOINT / PRIYAN_TOKENIZER / PRIYAN_DATASET to serve a trained main model. The UI's preset selector selects architecture/training configuration; inference uses the explicit server environment, never silently switches weights.

UI: overview, continuation workspace with SSE streaming/copy/download/session export, dataset manifest/filter report, architecture inspector, loss chart/start/stop/resume controls, evaluation report/export, hardware/checkpoints, generation/API settings. Corpus registration, preprocessing and filesystem paths are CLI/YAML operations; the UI does not claim to have a source-upload pipeline or full Monaco IDE.
