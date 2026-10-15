# PRIYAN CODE AI

An original PyTorch coding-LLM research project: random-initialized decoder Transformer, original Byte-Level BPE, next-token training, checksummed resume, KV-cached inference, local FastAPI service and React/TypeScript research dashboard. No pretrained weights or commercial generation APIs.

**Delivery status: verified local research implementation, not a production-quality trained coding assistant.** The included 115,008-parameter smoke checkpoint learned from a five-file original fixture for 60 CPU steps. It does not generate useful code. The 22,909,824 and 134,105,856-parameter presets both passed short forward/backward tests; neither underwent corpus pretraining. Read [BUILD_STATUS.md](BUILD_STATUS.md) for measured results and incomplete master-prompt requirements.

## Windows / WSL2 quick start

Use Python 3.12 and Node 22+. Extract this ZIP, open a terminal **inside PRIYAN-CODE-AI**, then:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_environment.py
python -m pytest -q
python scripts/smoke_test.py
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Open http://127.0.0.1:8000/docs for API documentation. Start with **Coding workspace** to stream predictions from the included smoke weights. Output is deliberately not replaced with fake AI answers.

Linux/WSL: `bash scripts/setup.sh`, `source .venv/bin/activate`, then the same Python commands. Tested on Linux Python 3.12 with CPU PyTorch; native Windows and CUDA were not executed here. A CPU install is sufficient for the included smoke workflow. For NVIDIA training select and install an appropriate CUDA wheel through the official [PyTorch installer](https://pytorch.org/get-started/locally/), rerun the hardware report, then use `python -m src.training.probe --model configs/model_23m.yaml`. Never infer CUDA readiness solely from an NVIDIA device name.

## Reproduce the fixture

```bash
python -m src.data.collect --config configs/dataset.yaml
python -m src.tokenizer.train_tokenizer --config configs/dataset.yaml
python -m src.data.prepare --config configs/dataset.yaml
python -m src.training.train --config configs/train_smoke.yaml --resume checkpoints/latest
python -m src.evaluation.evaluate --checkpoint checkpoints/latest
python -m src.inference.generate --checkpoint checkpoints/latest --prompt "def fibonacci(n):" --max-new-tokens 24
python -m src.inference.export --checkpoint checkpoints/latest
```

The shipped run already completed 60 steps, so resume at the same target is an intentional no-op. To create a fresh run, copy the training config and choose a **new checkpoint_dir and log**. Never erase or overwrite an experiment implicitly. To continue for more steps use a new experiment initialized from the old model deliberately; exact resume rejects a changed schedule. Source ingestion and token preparation are deterministic and reuse a matching completed manifest. Interrupted preparation rebuilds incomplete output; it does not yet resume at a record offset.

## Train the real 23M / 134M models

1. Copy `configs/dataset_production.yaml`. Replace placeholder sources with authorized local repositories, actual license/provenance evidence, and set `approved: true` only after checking permission. Use distinct repositories for validation/test, or deterministic repository hashing. Ingestion never fetches repositories or accepts private credentials.
2. Collect and train the **32,000-target vocabulary** with that production config. The tiny fixture vocabulary target is 512 (actual learned size 446) and is not appropriate for production pretraining. All supported byte values are included; a small corpus may yield fewer than 32,000 actual tokens. Record actual size. Once saved, the tokenizer is frozen; use a new directory to change its corpus/vocabulary.
3. Prepare shards. Inspect language distribution and held-out split sizes. The local exact/Jaccard deduper is a pilot implementation; use a scalable external deduper before massive ingestion.
4. Run the GPU probe. Edit microbatch/context after actual measurement. Keep context ≤ model context. Test 23M first, then initialize 134M from scratch with its own configuration. Defaults train about 20M/200M **processed** tokens, not guaranteed unique tokens. Do not repeatedly loop a tiny corpus to claim a large dataset.
5. Run the selected command, evaluate held-out losses, and review quality before increasing compute:

```bash
python -m src.data.collect --config configs/dataset_production.yaml
python -m src.tokenizer.train_tokenizer --config configs/dataset_production.yaml
python -m src.data.prepare --config configs/dataset_production.yaml
python -m src.data.inspect --dataset data/production/tokenized
python -m src.training.probe --model configs/model_23m.yaml
python -m src.training.train --config configs/train_23m.yaml
python -m src.training.train --config configs/train_134m.yaml
python -m src.training.train --config configs/train_134m.yaml --resume checkpoints/134m/latest
python -m src.evaluation.evaluate --checkpoint checkpoints/134m/latest --tokenizer artifacts/production/tokenizer.json --dataset data/production/tokenized
```

Choose one training run at a time. To serve the main model, set `PRIYAN_CHECKPOINT=checkpoints/134m/latest`, `PRIYAN_TOKENIZER=artifacts/production/tokenizer.json`, and `PRIYAN_DATASET=data/production/tokenized` in the backend process environment. Default serving uses the smoke checkpoint. Every generation request loads real weights; CPU inference is serialized and loading per request prioritizes simplicity over latency.

## Storage and reproducibility

No artificial 50 GB application limit. Dataset, tokenizer, checkpoint and log paths are editable in YAML; Windows drive paths or WSL mounted drives work through normal filesystem paths. Compact uint16 shards are used up to vocabulary ID 65535; larger vocabularies use uint32. At 200M stored tokens the raw token payload is about 400 MB (decimal), excluding source, metadata, checkpoints and dependencies. A 134M FP32 weight copy is ~536 MB; AdamW training checkpoints need several weight copies, temporary atomic-save space and retention headroom. Actual disk checks run before shard/checkpoint writes. No source data is automatically deleted. Checkpoint retention removes only project-generated `step-*.pt` files in the configured experiment directory.

Checkpoints contain trusted-local PyTorch pickle optimizer/RNG state. Never load an untrusted checkpoint. SHA-256 detects accidental corruption; it does not authenticate an attacker-controlled checkpoint and checksum together. Safetensors inference export is available separately.

## Project navigation

- `src/model`: RMSNorm, RoPE, causal SDPA, SwiGLU, residual blocks and optional KV cache.
- `src/data`, `src/tokenizer`: authorized ingestion, filters/dedupe, repository splits, original BPE and memmapped shards.
- `src/training`: AdamW, accumulation, mixed precision, cosine/warmup, gradient checkpointing, resumable RNG/sampler and assistant-only SFT.
- `src/inference`, `src/evaluation`: actual prediction, streaming, sampling, test loss/perplexity, syntax inspection, exports.
- `backend`, `frontend`: local service, process controls and eight research views.
- `src/agent`: bounded repository reads, proposed unified diffs, optional Docker-only test runner. No autonomous agent claim.
- `reports`: actual test/training verification evidence; `docs`: operating guidance.

## Tests

```bash
python -m pytest -q
cd frontend
npm ci
npm test
npm run build
```

The ZIP intentionally includes tiny smoke artifacts for immediate offline model loading after dependencies are installed. Runtime directories are excluded by `.gitignore`; do not commit your real corpora or large model weights. Internet is required only to install dependencies unless you add optional infrastructure yourself.
"# CODEFIRE" 
