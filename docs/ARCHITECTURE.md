# Architecture

Embedding → N pre-normalized blocks → RMSNorm → vocabulary projection. Each block uses joint bias-free QKV, rotary pairs, causal scaled-dot-product attention, a residual connection, then RMSNorm and `down(silu(gate(x))*up(x))` with a second residual. No pretrained weights. Weight initialization is normal std 0.02.

The head dimension must be even. RoPE offsets incorporate cache length. Prefill uses causal SDPA; multi-token cache extension uses an explicit absolute-position causal mask. Training compares logits at input position t against token t+1. SFT shifts labels the same way and masks prompt positions with -100.

Exact counts: 23M preset = 22,909,824 tied parameters; 134M = 134,105,856 untied parameters; smoke = 115,008. Tests initialize both large presets and backpropagate a three-token sequence. That verifies tensor/gradient mechanics, not GPU fit or training convergence.

Single-process training only. DDP, asynchronous data prefetch and distributed checkpointing are not implemented. The pretrained model has a finite learned context; exceeding it produces an explicit error rather than silently cropping the prompt.
