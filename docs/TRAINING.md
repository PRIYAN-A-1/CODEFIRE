# Training

Use `python scripts/check_environment.py`, then `python -m src.training.probe --model configs/model_23m.yaml` on your actual NVIDIA machine. The GPU probe performs real mixed-precision forward/backward/AdamW steps for candidate contexts 128/256/512 and microbatches 1/2/4, reporting peak allocation and 20% headroom. It does not auto-change a config. It runs in a fresh process. OOM can still occur later from other GPU users or fragmentation. No CUDA result was measured in this delivery environment.

Train with a versioned YAML config. Precision `auto` chooses BF16 on supported CUDA, FP16 + GradScaler otherwise, FP32 on CPU. The loop applies accumulation, clipping, AdamW, linear warmup then cosine decay, and validation at an independent sampler. Token throughput is a local observed step metric. CUDA timing in the training loop includes the synchronizations implicit in loss.item(); use a profiler for detailed kernel timing.

Complete checkpoints include config, model/optimizer/scaler, schedule position, tokens, Python/NumPy/Torch/CUDA RNG, data sampling generator, validation sampling generator and dataset/tokenizer checksums. Saves write a temporary file, fsync, replace, checksum, then update latest. A crash between checkpoint replacement and sidecar write may leave a new orphan without checksum; latest still points to a previous complete checkpoint when possible. Corrupt/missing state fails explicitly. Only trust checkpoints you created.

SIGINT/SIGTERM request stopping after a complete optimizer step; the completed state is saved. OOM preserves the previous complete checkpoint and exits with diagnostics. A hard kill cannot save in-progress gradients. Do not change sequence length, accumulation, batch size, LR, steps, tokenizer, architecture or dataset during exact resume. To alter schedules start a deliberately new experiment, optionally using the old model as initialization.

The shipped smoke run processed 3,840 token exposures drawn repeatedly from 210 stored train tokens. It is an intentional plumbing/overfit test. High held-out perplexity confirms poor generalization. Do not confuse processed exposures with corpus size. Budget enough genuinely diverse authorized data and GPU time for the 20M/200M pilot presets; quality is not guaranteed by token budget.

## Instruction training

Use approved JSONL records with instruction, response, approved=true. Start from your own trained base checkpoint:

```bash
python -m src.training.sft --checkpoint checkpoints/23m/latest --tokenizer artifacts/production/tokenizer.json --data instructions.jsonl --output checkpoints/sft/step-00000001.pt --epochs 1
```

SFT masks all user/prompt labels and trains assistant response plus end/EOS. It is a minimal full-precision single-example trainer, separate from the pretraining resume engine; SFT optimizer-resume, validation and before/after instruction evaluation are not integrated. Do not call its output instruction-capable until measured.
