# Troubleshooting

- Model not trained/loaded: run smoke_test.py or point environment variables at your completed checkpoint and matching tokenizer.
- No train/validation sequences: add enough genuinely distinct authorized repositories for both splits or reduce smoke context. Never copy train records into validation.
- Tokenizer is frozen/dataset differs: use new artifact/output paths for changed corpora. Do not overwrite the tokenizer of a trained model.
- Resume mismatch: use the exact original YAML and dataset. Changing steps changes the cosine schedule and is intentionally rejected.
- CUDA unavailable: install a compatible PyTorch CUDA wheel using the official installer; verify drivers inside the execution OS, including WSL.
- CUDA OOM: reduce microbatch/context in a new experiment after a fresh-process probe. Accumulate smaller batches rather than assuming the GPU capacity.
- Insufficient disk: relocate dataset/checkpoint directories to another drive. Do not manually delete unknown user files. Leave headroom for atomic saves plus two retained checkpoints.
- PowerShell cannot activate: run .venv/Scripts/python.exe directly; activation is optional.
- Dashboard cannot connect: start API on port 8000 and Vite on 5173, check errors/authentication. Use one API worker.
- Poor code output: expected for smoke weights; verify real corpus quality, held-out loss and sufficient GPU training. Do not hide this with hardcoded answers.
- Training start returns conflict: an experiment exists, or model resources are in use. Resume matching preset or use a new config via CLI.
