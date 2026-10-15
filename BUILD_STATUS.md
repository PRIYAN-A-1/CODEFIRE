# Build status — PRIYAN CODE AI v0.1.0

This is a real, tested local LLM research build. **The main coding model has not been pretrained.** The included smoke weights do not produce useful code. No third-party pretrained weights or generation APIs were used.

## Executed evidence

| Check | Result |
|---|---|
| Python tests | 43 passed, including both model presets' short forward/backward, causal masking, KV cache, tokenizer, exact resume, API/auth/real generation |
| Frontend component tests | 2 passed |
| React TypeScript + Vite production build | Passed |
| Original BPE | Train split only; 446 actual tokens, 512 target, byte alphabet preserved |
| Smoke training | 115,008 parameters, 60 CPU optimizer steps, 3,840 processed token exposures |
| Stored corpus | 5 original code files; train 210 / validation 189 / test 144 tokens |
| Train loss | 6.247012 at step 1 → 3.029353 at step 60 |
| Last single-batch validation loss | 6.476296 |
| Held-out test fixture loss/perplexity | 6.489582 / 658.248 |
| Python syntax sample check | 0 of 2 valid completions; failures retained |
| Actual HTTP API generation + SSE | Passed through uvicorn and real checkpoint |
| CLI resume + data inspection | Passed |
| Assistant-only SFT CLI | One original approved example trained for one step; no instruction-quality claim |
| Safetensors export | Executed successfully |
| CUDA | Unavailable; batch/VRAM probe reports unavailable |
| Browser visual test | Not run: Chromium installation returned invalid download bytes |
| Native Windows / WSL / GPU | Instructions supplied, not executed here |

Reports are in `reports/`, `logs/train.jsonl`, and `evaluations/`. These are fixture measurements, not coding benchmark scores. Throughput is this environment's measurement and does not predict laptop performance.

## Requirement coverage and practical limits

| Area | Implemented | Remaining |
|---|---|---|
| Model | RMSNorm, RoPE, SwiGLU, SDPA causal attention, random init, KV cache; 22,909,824 / 134,105,856 parameter presets | Full 23M/134M corpus pretraining and quality evaluation |
| Tokenizer | Original byte BPE, whitespace/Unicode tests, special tokens, checksum, freeze guard | Corpus-scale training and tokenizer quality study |
| Corpus | Permission gates, SPDX allowlist, source provenance fields, local scanning, secret/PII heuristics, Python AST, exact/global Jaccard dedupe, repo splits, benchmark hash exclusions | Scalable MinHash/SQLite ingestion, tree-sitter adapters, full license/PII audit, benchmark fuzzy matching, source removal propagation |
| Storage | Configurable paths on any mounted drive, physical-space guards, uint16/uint32 shards, memmap, checksum manifests, checkpoint retention | Direct remote object-store streaming, source download adapters, resumable record-offset ingestion; collection is RAM-bound |
| Training | CPU/CUDA path, auto precision, accumulation, clipping, warmup/cosine, gradient checkpointing, actual GPU probe implementation | GPU validation, DDP, TensorBoard, prefetch, automatic VRAM-tuned config selection |
| Recovery | Atomic checkpoint/sidecar/latest writes, optimizer/scaler/RNG/sampler state, integrity checks, exact resume, graceful stop | Transactional multi-file saves and cross-version checkpoint migration; OOM restores previous completed save |
| Inference | Greedy/temperature/top-k/top-p, repetition penalty, seeded generation, EOS/context bounds, streaming | Deployment optimization, inference loading directly from safetensors export |
| Instruction tuning | Actual own-model SFT, prompt masking, assistant/end/EOS training | Dedicated SFT resume/validation and before/after functional studies |
| Evaluation | Real held-out CE/perplexity, fixture throughput, syntax validity, JSON/Markdown failure reports | Isolated compilation/unit-test/pass@1/repair scoring and public benchmark integrations; these fields stay null |
| Backend | Typed local API, streamed generation, resource lock, allowlisted job start/stop/resume, hardware/dataset/checkpoints/status | Robust external supervisor, multiworker/multiuser job persistence, hardened remote hosting |
| Dashboard | Eight functional views, real metrics, SSE continuation, copy/download/session export, loss chart, presets, job controls, reports/settings | Monaco editor, source registration/ingestion UI, experiment compare/advanced analytics |
| Agent | Confined reads, secret gate, proposed unified diff, optional Docker-only test function | Autonomous planner/repair loop and Docker execution verification; no autonomous-agent claim |
| MLOps | SHA-256 artifacts, saved configs, model-card export and reproducibility manifest | Full version registry, git-linked promotion workflow, production deployment |

## Run next on your laptop

1. Follow README installation. Verify actual GPU/VRAM; no GPU details were inferred from your laptop description.
2. Run the included smoke workflow before provisioning a full corpus.
3. Configure real authorized repositories in `configs/dataset_production.yaml`, train/freeze your production tokenizer, create isolated validation/test data.
4. Probe and train `train_23m.yaml`, inspect held-out quality, then train `train_134m.yaml` from random initialization.
5. Perform instruction tuning and isolated functional evaluation before using this as a coding assistant.

No artificial 50 GB cap is enforced. Use configurable paths on another mounted drive when physical disk is insufficient. This package does not promise ChatGPT-level coding quality from a 134M model.
