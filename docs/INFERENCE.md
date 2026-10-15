# Inference

`python -m src.inference.generate --checkpoint checkpoints/latest --prompt "def fibonacci(n):" --max-new-tokens 24`

The Engine verifies checkpoint/tokenizer hashes, reconstructs the original architecture, loads its weights and samples actual autoregressive tokens with KV caching. Temperature zero is greedy; otherwise top-k and nucleus sampling apply. A local seeded generator makes runs reproducible within the tested environment. Repetition penalty is available in Python. EOS stops; output limits and context are enforced. Invalid/partial Unicode bytes are buffered during streaming where possible.

`python -m src.inference.export` writes standalone safetensors weights, architecture JSON and a small model-card JSON. The live Engine currently loads resumable local checkpoints, not safetensors exports; the export is for external deployment integration. Tied weights are cloned during export to avoid shared-storage errors. Copy the matching tokenizer and retain attribution/provenance when distributing a model.

Raw base models perform continuation. SFT uses user/assistant/end framing. The current dashboard defaults to continuation rather than pretending an untrained model is a chat assistant. Small models and inadequate training produce invalid or incorrect code; no generated code is automatically executed.
