# MASTER BUILD PROMPT — PRIYAN CODE AI
## Build a Complete, Original, Multilingual Coding LLM From Scratch

### YOUR ROLE

Act as a team of:

- Principal AI Research Scientist
- Large Language Model Architect
- Senior PyTorch Engineer
- GPU Optimization Engineer
- AI Dataset Engineer
- Distributed Training Engineer
- Senior Python Backend Engineer
- React/TypeScript Frontend Engineer
- DevOps and MLOps Engineer
- AI Safety and Security Engineer
- Software Quality Assurance Engineer

Your mission is to **design, implement, train, evaluate, and package a real Coding Large Language Model named PRIYAN CODE AI**.

This must be a functional software engineering project, not a conceptual demonstration, fake AI interface, static mockup, or API wrapper around another commercial model.

Implement actual source code, configuration, dataset processing, model architecture, training logic, inference, API, frontend, automated tests, and documentation.

Do not stop at generating a folder structure or writing TODO comments. Implement the required functionality in executable form.

If a computationally expensive training operation cannot be completed in the available environment, implement and validate the training pipeline using a small smoke test, preserve resumable checkpoints, and document the exact steps to run the larger training job.

Never claim that a model has been trained, benchmarks passed, or features work unless they were actually executed and verified.

---

# 1. PROJECT IDENTITY

Project Name: PRIYAN CODE AI

Primary Model: PRIYAN-CODE-134M

Research Model: PRIYAN-CODE-23M

Category: Large Language Model / Generative AI / Code Intelligence

Training Method: **Pretraining from random initialization**

Primary Goal: Build an original, multilingual, decoder-only Transformer trained on programming code.

Secondary Goal: Build an instruction-following coding assistant using the original pretrained model.

Long-Term Goal: Extend the assistant into a repository-aware autonomous coding agent.

Branding:

- Product: PRIYAN CODE AI
- Main interface title: PRIYAN CODE AI
- Model family: PRIYAN-CODE
- Design: Premium, modern developer workspace
- Theme: Dark, minimalist, professional, cyan/blue accents

## Non-Negotiable Model Requirements

1. Do not download or reuse pretrained model weights.
2. Do not fine-tune Qwen, Llama, DeepSeek, GPT, StarCoder, or any existing pretrained LLM.
3. Implement the Transformer architecture in PyTorch.
4. Initialize all trainable model parameters from scratch.
5. Train an original coding tokenizer.
6. Implement real next-token pretraining.
7. Build a genuine model checkpointing system.
8. Provide actual autoregressive inference.
9. Add proper evaluation and automated tests.
10. Keep model training, data preparation and inference reproducible.

Using established software libraries such as PyTorch, Hugging Face Tokenizers, NumPy and FastAPI is permitted. The prohibition concerns pretrained model parameters, not the use of engineering libraries.

---

# 2. HARDWARE & ENVIRONMENT

Known hardware:

- Operating System: Windows
- System RAM: 32 GB or higher
- GPU: NVIDIA
- Free SSD space initially reported: approximately 50 GB
- Exact NVIDIA GPU and VRAM: Unknown

## Dynamic Hardware Detection

Implement a hardware profiler that detects:

- NVIDIA GPU model
- GPU VRAM
- CUDA availability
- CUDA and PyTorch versions
- CPU cores
- Available system RAM
- Available local storage
- Optional external storage paths
- Optional cloud storage configuration

Generate a report such as:

```text
PRIYAN CODE AI — HARDWARE REPORT

GPU: [Detected NVIDIA GPU]
VRAM: [Actual detected value]
System RAM: [Detected]
CUDA: [Detected status]
Available disk: [Detected]

Recommended Model: [23M/134M]
Recommended Precision: [BF16/FP16/FP32]
Recommended Batch Size: [Measured]
Recommended Sequence Length: [Measured]
Estimated Checkpoint Size: [Calculated]
```

Never invent detected hardware specifications.

## Storage Requirements

Do not impose an artificial 50 GB application-level storage limit.

Implement support for:

- Local SSD
- Secondary internal drives
- External SSD/HDD
- Configurable dataset directory
- Configurable cache directory
- Configurable checkpoint directory
- Remote object storage
- Streaming datasets
- Dataset sharding
- Memory-mapped token files
- Checkpoint retention policies
- Disk-space monitoring

Users must configure paths and budgets.

Before downloading data or saving checkpoints, check available disk space.

If available storage becomes insufficient, pause safely, preserve recoverable state where possible, and present cleanup or relocation options.

Never delete original source data or user files automatically.

---

# 3. TECHNOLOGY STACK

## Machine Learning

- Python 3.11 or compatible stable version
- PyTorch
- CUDA
- Hugging Face Tokenizers
- NumPy
- PyArrow
- Safetensors
- TensorBoard
- YAML configuration

## Dataset Processing

- Python streaming iterators
- GitPython or controlled Git subprocess operations
- Tree-sitter for supported languages
- SHA-256 hashing
- MinHash or equivalent near-duplicate detection
- SQLite metadata tracking
- Parquet/JSONL for metadata
- Binary token shards for training

## Backend

- FastAPI
- Pydantic
- Uvicorn
- Server-Sent Events or WebSockets
- Asynchronous job-status reporting

## Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- Monaco Editor
- Lucide React
- Motion for React where useful

## Development

- Git
- Ruff
- Pytest
- Mypy where practical
- Pre-commit hooks
- Docker for optional deployment
- GitHub Actions for CI

Pin tested dependency versions. Do not rely blindly on untested latest versions.

---

# 4. TWO-MODEL TRAINING STRATEGY

Implement two configurations.

## Model A — PRIYAN-CODE-23M

Purpose: Validate the architecture and entire pipeline before expensive training.

Suggested configuration:

```yaml
name: PRIYAN-CODE-23M
architecture: decoder_transformer
vocab_size: 32000
hidden_size: 384
num_layers: 6
num_attention_heads: 6
intermediate_size: 1024
max_context_length: 512
normalization: rmsnorm
activation: swiglu
position_encoding: rope
tie_word_embeddings: true
```

Target approximately 23 million parameters.

Calculate and print the exact parameter count from the implemented architecture.

This model must support:

- Forward pass
- Backpropagation
- Short-context training
- Loss calculation
- Checkpoint save/load
- Autoregressive token generation

Complete this model before scaling to 134M.

## Model B — PRIYAN-CODE-134M

Purpose: Main scratch-pretrained Coding LLM.

Configuration:

```yaml
name: PRIYAN-CODE-134M

architecture: decoder_only_transformer

vocab_size: 32000
hidden_size: 768
num_layers: 12
num_attention_heads: 12
head_dim: 64

intermediate_size: 2048

activation: swiglu
normalization: rmsnorm
position_encoding: rope

max_context_length: 1024
rope_theta: 10000

tie_word_embeddings: false
attention_type: causal

initialization: random
training_objective: causal_language_modeling
```

Target approximately 134 million parameters.

The final parameter count must be calculated programmatically and recorded.

Use:

- Causal self-attention
- Query, key and value projection
- Rotary position embeddings
- RMSNorm
- SwiGLU feed-forward networks
- Residual connections
- Final normalization
- Vocabulary projection
- Cross-entropy loss

Use PyTorch scaled dot-product attention where supported.

Implement correctly functioning causal masks and compatible key-value caching for inference.

Avoid unnecessary intermediate attention tensors.

---

# 5. ORIGINAL TOKENIZER ENGINE

Develop an original code-oriented Byte-Level BPE tokenizer.

Vocabulary target: 32,000 tokens.

Supported inputs:

- Python
- JavaScript
- TypeScript
- C
- C++
- Java
- Go
- Rust
- SQL
- Bash
- JSON
- YAML
- HTML/CSS
- Other languages as the dataset grows

Special tokens:

```text
<unk>
<|bos|>
<|eos|>
<|pad|>

<|fim_prefix|>
<|fim_middle|>
<|fim_suffix|>

<|system|>
<|user|>
<|assistant|>
<|end|>
```

Requirements:

- Preserve source code indentation.
- Preserve whitespace and line breaks.
- Preserve punctuation and symbols.
- Support Unicode.
- Handle uncommon identifiers.
- Encode and decode source code correctly.
- Provide token count estimates.
- Store tokenizer version and checksum.
- Add tests for round-trip correctness.
- Freeze the tokenizer vocabulary before large-scale pretraining.

Build:

```text
src/tokenizer/
  train_tokenizer.py
  encode.py
  decode.py
  tokenizer_utils.py
```

Deliver:

```text
artifacts/tokenizer.json
artifacts/tokenizer_config.json
artifacts/tokenizer_report.json
```

Never train on artificially duplicated data merely to reach a vocabulary target.

---

# 6. MULTILINGUAL CODING DATASET ENGINE

Build a real dataset creation and management system.

Do not use a fabricated dataset that claims to include millions of samples.

The platform must acquire, validate, filter, tokenize, version and stream legitimate source-code records.

## Supported Languages

Tier 1:

- Python
- JavaScript
- TypeScript
- C
- C++
- Java

Tier 2:

- Go
- Rust
- C#
- SQL
- Bash
- PHP

Tier 3:

- Kotlin
- Swift
- Dart
- Ruby
- R
- HTML
- CSS
- JSON
- YAML
- Dockerfile

Make language support extensible through adapters.

## Data Sources

Use sources with appropriate training rights.

Possible sources include:

1. User-owned original repositories.
2. Explicitly approved permissively licensed projects.
3. Datasets with verified training permissions.
4. Original code-generation tasks.
5. Original unit tests and debugging examples.
6. Code documentation with permitted reuse.

Support The Stack v2 only when the user has accepted its access requirements, and implement compliance with source licensing and removal requirements.

Never bypass dataset access controls.

Do not assume all publicly accessible GitHub repositories are freely reusable.

Do not scrape private repositories or restricted content.

## Dataset Schema

Each record should contain:

```json
{
  "id": "unique-id",
  "language": "python",
  "content": "source-code-content",
  "source_type": "git",
  "repository": "repository-name",
  "source_url": "repository-url",
  "commit_sha": "commit-hash",
  "path": "relative/path.py",
  "license_spdx": "MIT",
  "content_sha256": "hash",
  "token_count": 512,
  "split": "train",
  "quality_status": "validated"
}
```

Preserve data provenance, source attribution, and applicable license requirements.

## Dataset Cleaning

Implement:

- Exact deduplication
- Near-duplicate detection
- Generated-code filtering
- Vendored dependency filtering
- Binary-file removal
- Oversized/minified-file filtering
- Empty-file filtering
- Secret detection
- Credential removal
- Personal-data minimization
- License filtering
- Known benchmark contamination checks
- Language detection
- Syntax parsing where practical
- Source quality reporting

Do not silently mutate licensed source files in a way that destroys provenance.

## Train / Validation / Test Split

Use repository-aware and near-duplicate-aware splitting.

Keep held-out repositories separate from training repositories where feasible.

Prevent test-set leakage.

Document the split rules.

## Dataset Budget

Support multiple scaling stages:

```yaml
dataset:
  token_targets:
    smoke_test: 1000000
    small_model: 20000000
    main_pilot: 200000000
    extended: 1000000000
    optional_large_scale: 2000000000

  streaming: true
  sharded_storage: true
  token_dtype: uint16

  max_local_storage_gb: null
  external_storage_enabled: true
  remote_storage_enabled: true
```

A null maximum storage setting means no artificial application limit, not unlimited physical disk space.

Use uint16 only while vocabulary IDs fit its representable range.

Provide a configurable language distribution.

Avoid inflating dataset size through uncontrolled repetition.

---

# 7. DATASET PIPELINE ARCHITECTURE

Implement a complete pipeline:

```text
DATA SOURCE
    ↓
LICENSE & PERMISSION VALIDATION
    ↓
METADATA EXTRACTION
    ↓
LANGUAGE DETECTION
    ↓
SECRET / SENSITIVE DATA FILTERING
    ↓
QUALITY FILTERING
    ↓
DEDUPLICATION
    ↓
TRAIN / VALIDATION / TEST SPLIT
    ↓
CODE TOKENIZATION
    ↓
TOKEN SHARDS
    ↓
TRAINING DATA LOADER
```

Requirements:

- Configurable parallel processing
- Deterministic shuffling
- Reproducible random seeds
- Iterator-based streaming
- Incremental ingestion
- Crash recovery
- Progress logging
- Dataset manifests
- Content hashes
- Duplicate statistics
- Token statistics
- Language distribution analytics
- Storage consumption reporting

Store token shards in a compact binary format.

Use memory-mapped reading where appropriate.

Support resumable dataset preparation.

Provide commands:

```bash
python -m src.data.collect --config configs/dataset.yaml

python -m src.data.prepare --config configs/dataset.yaml

python -m src.data.inspect --dataset data/tokenized
```

Every command must be implemented and tested.

---

# 8. GPU TRAINING ENGINE

Build an original, configurable PyTorch training engine.

The system must support:

- CUDA training
- CPU smoke testing
- Mixed precision
- BF16 when supported
- FP16 with gradient scaling when required
- FP32 fallback
- Gradient accumulation
- Gradient checkpointing
- AdamW optimizer
- Weight decay
- Learning-rate warmup
- Cosine learning-rate scheduling
- Gradient clipping
- Configurable effective batch size
- Deterministic seeds where practical
- Training metrics
- Validation
- Recoverable checkpoints
- Out-of-memory diagnostics

## Baseline Training Settings

```yaml
training:
  model: PRIYAN-CODE-134M

  optimizer: adamw

  learning_rate: 0.0003
  min_learning_rate: 0.00003

  betas: [0.9, 0.95]
  weight_decay: 0.1

  scheduler: cosine
  warmup_ratio: 0.02

  sequence_length: 512
  micro_batch_size: 1
  gradient_accumulation_steps: 32

  grad_clip_norm: 1.0

  mixed_precision: auto
  gradient_checkpointing: true

  target_training_tokens: 200000000

  eval_every_steps: 500
  checkpoint_every_steps: 2000

  checkpoint_keep_last: 2

  seed: 42
```

Treat these values as starting hyperparameters, not guaranteed optimal settings.

## Adaptive VRAM Profiler

Before training:

1. Detect GPU VRAM.
2. Check CUDA capabilities.
3. Run a safe allocation/forward/backward probe.
4. Estimate memory use.
5. Select a supported microbatch and sequence length.
6. Report estimated effective batch size.
7. Reserve practical memory headroom.
8. Fail safely if the configuration cannot fit.

Do not make unsupported claims that a specific VRAM capacity guarantees success.

## Training Stages

Stage A: Run a minimal forward/backward test.

Stage B: Overfit a tiny batch to verify optimization.

Stage C: Train PRIYAN-CODE-23M on a small dataset.

Stage D: Evaluate and debug the complete pipeline.

Stage E: Initialize PRIYAN-CODE-134M from random weights.

Stage F: Execute the 200M-token pilot.

Stage G: Analyze validation metrics and code quality.

Stage H: Continue toward 1B tokens only when compute, storage and validation justify it.

Add optional cloud GPU training through the same configuration interface.

Support additional GPU resources through a future distributed-training adapter without making the single-GPU implementation unnecessarily complex.

---

# 9. CHECKPOINT & RECOVERY SYSTEM

The training engine must save:

- Model weights
- Optimizer state
- Scheduler state
- Gradient scaler state when applicable
- Training step
- Number of processed tokens
- Random state
- Dataset position/shuffle state where possible
- Training configuration
- Model configuration
- Tokenizer checksum
- Dataset manifest reference

Implement:

- Atomic checkpoint save
- Integrity verification
- Resume training
- Checkpoint rotation
- Disk-space checks
- Graceful interruption handling
- Checkpoint migration/version compatibility

Checkpoint corruption must produce an explicit error.

Never silently restart training from step zero after a failed resume.

Provide:

```bash
python -m src.training.train \
  --config configs/train_134m.yaml

python -m src.training.train \
  --config configs/train_134m.yaml \
  --resume checkpoints/latest
```

Both commands must correspond to actual implemented entrypoints.

---

# 10. CODE GENERATION / INFERENCE ENGINE

Build a real autoregressive generation engine.

Support:

- Prompt tokenization
- Model forward inference
- KV caching
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p sampling
- Maximum generation length
- EOS stopping
- Repetition controls
- Seeded generation
- Streaming generated tokens

Example API:

```python
result = model.generate(
    prompt="def fibonacci(n):",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)
```

The real implementation may use an inference wrapper rather than a method directly attached to the neural model.

The model must load its own weights and tokenizer.

Do not substitute an external AI API.

Support CLI:

```bash
python -m src.inference.generate \
  --checkpoint checkpoints/latest \
  --prompt "def fibonacci(n):" \
  --max-new-tokens 256
```

During early raw pretraining, support code continuation as the primary capability.

Do not promise reliable instruction-following until instruction training has been performed and evaluated.

---

# 11. INSTRUCTION-FOLLOWING TRAINING

After base pretraining works, implement instruction supervised fine-tuning using the original model weights.

This is a continuation of our own scratch-trained model, not fine-tuning a third-party foundation model.

Instruction dataset categories:

1. Natural-language-to-code
2. Code explanation
3. Bug identification
4. Bug repair
5. Refactoring
6. Unit-test generation
7. Algorithm implementation
8. Code translation
9. API implementation
10. Structured programming tasks

Example:

```json
{
  "instruction": "Write a Python function to reverse a string.",
  "response": "def reverse_string(text):\n    return text[::-1]",
  "language": "python",
  "task_type": "code_generation",
  "tests": [
    "assert reverse_string('abc') == 'cba'"
  ]
}
```

Requirements:

- Instruction conversation formatting
- Loss masking for non-target prompt tokens
- Correct special token usage
- EOS training
- Supervised instruction loss
- Held-out evaluation
- Before-and-after comparisons

Do not rely entirely on unverified synthetic examples.

Generate original examples and validate solutions with real tests whenever possible.

---

# 12. AUTOMATED EVALUATION ENGINE

Evaluate model quality using actual metrics.

Required metrics:

- Training cross-entropy
- Validation cross-entropy
- Perplexity
- Tokens per second
- GPU memory usage
- Training step time
- Syntax validity
- Compilation success
- Unit-test pass rate
- pass@1 where appropriate
- Bug-fix success rate
- Inference latency

Use compatible evaluation tasks such as EvalPlus and LiveCodeBench where permitted and appropriate.

Keep evaluation data isolated.

Do not train on benchmark solutions or intentionally contaminated examples.

Build evaluation reports in JSON and Markdown.

Example:

```json
{
  "model": "PRIYAN-CODE-134M",
  "checkpoint": "checkpoint-id",
  "training_tokens": 200000000,
  "validation_loss": null,
  "perplexity": null,
  "syntax_validity": null,
  "unit_test_pass_rate": null,
  "status": "not_evaluated"
}
```

Values must remain null or explicitly unavailable until they have been measured.

Include failure examples and weaknesses in evaluation reports.

---

# 13. CODING ASSISTANT BACKEND

Create a FastAPI backend.

Suggested endpoints:

```text
GET    /health
GET    /api/system/hardware
GET    /api/system/storage

GET    /api/models
GET    /api/models/{model_id}

POST   /api/generate
POST   /api/generate/stream

GET    /api/datasets
POST   /api/datasets/validate

GET    /api/training/status
POST   /api/training/start
POST   /api/training/stop
POST   /api/training/resume

GET    /api/evaluation/results
POST   /api/evaluation/run

GET    /api/checkpoints
```

Requirements:

- Pydantic request and response validation
- Safe error handling
- Structured logs
- Model loading status
- Training-job process management
- GPU resource conflict prevention
- Streaming inference
- Bounded request sizes
- Controlled concurrency
- Configurable authentication for remote deployments

Training must run as a managed job, not block ordinary HTTP request handlers.

Do not expose unrestricted training or shell endpoints to the public internet.

---

# 14. PROFESSIONAL WEB DASHBOARD

Build a complete premium web interface.

Design inspiration:

- VS Code
- GitHub Copilot
- Modern AI coding workspaces
- Research model monitoring dashboards

Do not copy another product's branded interface exactly.

Theme:

```text
Background: #070B14
Surface: #101827
Primary Accent: #00D4FF
Secondary Accent: #6366F1
Text: #F8FAFC
Muted Text: #94A3B8
```

Use responsive layouts and accessible contrast.

## PAGE 1 — OVERVIEW

Display:

- Project name
- Current model
- Parameters
- GPU status
- Training status
- Dataset token count
- Latest checkpoint
- Evaluation status
- Recent activities

Use real backend information.

## PAGE 2 — AI CODING WORKSPACE

Create:

- Chat interface
- Code editor
- Language selector
- Prompt input
- Generated code output
- Streaming token display
- Copy code
- Download code
- Save conversation
- Code syntax highlighting
- Basic error messages

Show model name and checkpoint currently loaded.

If no trained model exists, display a clear "Model not trained/loaded" state.

Never generate fake responses pretending to come from PRIYAN CODE AI.

## PAGE 3 — DATASET MANAGER

Display:

- Registered data sources
- Dataset sizes
- Languages
- Token counts
- License metadata
- Quality status
- Deduplication statistics
- Train/validation/test distribution
- Preparation status
- Disk usage

Controls:

- Register source
- Validate metadata
- Run preprocessing
- Inspect dataset
- Review rejected records

## PAGE 4 — MODEL ARCHITECTURE

Show:

- Parameter count
- Layers
- Attention heads
- Hidden size
- Context length
- Vocabulary size
- Memory estimates
- Model configuration

Optional interactive architecture visualization.

## PAGE 5 — TRAINING LAB

Display:

- Current training step
- Processed tokens
- Training loss
- Validation loss
- Learning rate
- Tokens per second
- GPU VRAM usage
- CPU/RAM usage
- Checkpoint history
- Training logs

Use real logged values, never fabricated progress.

Controls:

- Start experiment
- Pause safely
- Stop
- Resume checkpoint
- Open logs
- Compare experiments

## PAGE 6 — EVALUATION LAB

Display:

- Available evaluation suites
- Validation metrics
- Syntax test results
- Compile test results
- Unit-test scores
- Before/after comparisons
- Failure cases
- Exportable reports

## PAGE 7 — STORAGE & SYSTEM

Display:

- Available disk space
- Dataset paths
- Checkpoint paths
- Cache paths
- GPU information
- Storage allocations
- Optional remote storage status

Provide configurable storage paths.

## PAGE 8 — SETTINGS

Support:

- Model selection
- Inference temperature
- Maximum output tokens
- Training configuration
- Dataset settings
- Theme
- API configuration
- Authentication and security settings

---

# 15. FUTURE CODING AGENT MODULE

Implement a clearly separated experimental coding-agent layer.

Do not treat code generation alone as autonomous software engineering.

Agent components:

```text
Prompt
  ↓
Task Planner
  ↓
Repository Reader
  ↓
Coding LLM
  ↓
Patch Generator
  ↓
Human Approval
  ↓
Sandboxed Tool Runner
  ↓
Test Results
  ↓
Repair Proposal
```

Capabilities to implement progressively:

- Read project files
- Search repository
- Understand file structure
- Suggest modifications
- Produce unified diffs
- Run permitted tests in a sandbox
- Collect compiler errors
- Propose corrective patches
- Repeat bounded repair attempts

Security requirements:

- Sandbox command execution
- No unrestricted host shell access
- No automatic destructive commands
- No credential exposure
- Default-deny network access during code execution
- Timeout and resource limits
- File access restricted to workspace
- Path traversal prevention
- User approval for sensitive operations
- Audit logs

If the trained model cannot reliably perform advanced tasks, clearly report limitations instead of simulating agent success.

---

# 16. TESTING REQUIREMENTS

Build an automated test suite covering:

## Tokenizer Tests

- Encoding
- Decoding
- Round-trip preservation
- Whitespace
- Unicode
- Indentation
- Special tokens
- Unknown text

## Transformer Tests

- Correct tensor shapes
- Parameter count
- Causal masking
- RoPE application
- Attention output
- Gradient propagation
- Finite loss
- GPU execution

## Training Tests

- Training-step execution
- Gradient accumulation
- Optimizer stepping
- Scheduler stepping
- Mixed-precision behavior
- Checkpoint restoration
- Interrupted training recovery
- Tiny-dataset overfitting

## Dataset Tests

- Input validation
- License filtering
- Duplicate detection
- Split isolation
- Token shards
- Data loading
- Missing files
- Corrupted records

## Backend Tests

- API response contracts
- Invalid payloads
- Model-not-loaded errors
- Training status
- Safe failure behavior
- Authentication behavior

## Frontend Tests

- Rendering
- API loading states
- Error states
- Training status display
- Responsive layouts
- Basic accessibility

Run actual tests and report pass/fail counts.

Do not claim that unexecuted tests passed.

---

# 17. PERFORMANCE OPTIMIZATION

Optimize for consumer NVIDIA GPUs.

Implement:

- Memory-efficient attention
- Mixed precision
- Gradient accumulation
- Gradient checkpointing
- Data prefetching
- Pinned memory where useful
- Efficient binary datasets
- Limited checkpoint retention
- Safe batch-size tuning
- VRAM profiling
- Token throughput reporting

CPU-only support may be limited to smoke tests and small-model experiments.

Support larger GPU environments without changing core model code.

Do not claim a 134M model will perform comparably to multi-billion-parameter coding LLMs.

---

# 18. MODEL VERSIONING & MLOPS

Implement a model registry for local artifacts.

Each model version should have:

```text
model_name
model_version
architecture
parameter_count
tokenizer_version
dataset_version
training_configuration
training_tokens
checkpoint_path
validation_metrics
creation_time
git_commit
```

Features:

- List checkpoints
- Compare versions
- Restore versions
- Verify artifact hashes
- Export supported artifacts
- Generate model cards
- Track training configuration
- Preserve data provenance

Support Safetensors model-weight export.

Keep optimizer-resume checkpoints separate from lightweight inference exports.

---

# 19. FULL PROJECT STRUCTURE

Generate a structured repository:

```text
PRIYAN-CODE-AI/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
├── Makefile
├── Dockerfile
│
├── configs/
│   ├── model_23m.yaml
│   ├── model_134m.yaml
│   ├── train_23m.yaml
│   ├── train_134m.yaml
│   ├── dataset.yaml
│   └── inference.yaml
│
├── src/
│   ├── tokenizer/
│   │   ├── train_tokenizer.py
│   │   ├── encode.py
│   │   └── decode.py
│   │
│   ├── data/
│   │   ├── collect.py
│   │   ├── filter.py
│   │   ├── deduplicate.py
│   │   ├── split.py
│   │   ├── prepare.py
│   │   ├── inspect.py
│   │   └── dataset.py
│   │
│   ├── model/
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── rope.py
│   │   ├── attention.py
│   │   ├── rmsnorm.py
│   │   ├── mlp.py
│   │   ├── transformer_block.py
│   │   └── transformer.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── sft.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── checkpoint.py
│   │   └── metrics.py
│   │
│   ├── inference/
│   │   ├── generate.py
│   │   ├── sampler.py
│   │   └── model_loader.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── syntax_eval.py
│   │   ├── compile_eval.py
│   │   └── benchmark.py
│   │
│   ├── agent/
│   │   ├── planner.py
│   │   ├── repository.py
│   │   ├── patcher.py
│   │   ├── sandbox.py
│   │   └── test_runner.py
│   │
│   └── system/
│       ├── hardware.py
│       ├── storage.py
│       └── logging.py
│
├── backend/
│   ├── main.py
│   ├── api/
│   ├── services/
│   ├── schemas/
│   └── jobs/
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── styles/
│   └── public/
│
├── scripts/
│   ├── setup.sh
│   ├── setup.ps1
│   ├── check_environment.py
│   ├── smoke_test.py
│   ├── start_backend.sh
│   └── start_frontend.sh
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_dataset.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_api.py
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── tokenized/
│   └── manifests/
│
├── artifacts/
├── checkpoints/
├── logs/
├── evaluations/
│
└── docs/
    ├── ARCHITECTURE.md
    ├── INSTALLATION.md
    ├── DATASET.md
    ├── TRAINING.md
    ├── INFERENCE.md
    ├── API.md
    ├── SECURITY.md
    ├── BENCHMARKS.md
    └── TROUBLESHOOTING.md
```

All runtime-generated directories must be ignored by Git unless an artifact is intentionally versioned.

Do not commit large datasets, checkpoint weights, secrets or local environment files.

---

# 20. INSTALLATION AUTOMATION

Provide installation instructions for Windows + WSL2 Ubuntu.

Setup script must:

1. Validate Python version.
2. Detect NVIDIA GPU.
3. Detect CUDA-enabled PyTorch.
4. Install compatible dependencies.
5. Create required directories.
6. Initialize configuration.
7. Validate storage permissions.
8. Execute small smoke tests.
9. Display detailed setup status.

Use the officially supported PyTorch installation command appropriate for the detected CUDA environment.

Do not hardcode an incompatible CUDA wheel.

Provide:

```bash
python scripts/check_environment.py

python scripts/smoke_test.py

pytest tests/ -v
```

## Startup Commands

Implement a reproducible workflow:

```bash
# Train tokenizer
python -m src.tokenizer.train_tokenizer \
  --config configs/dataset.yaml

# Prepare dataset
python -m src.data.prepare \
  --config configs/dataset.yaml

# Run small model
python -m src.training.train \
  --config configs/train_23m.yaml

# Run main model
python -m src.training.train \
  --config configs/train_134m.yaml

# Run evaluation
python -m src.evaluation.evaluate \
  --checkpoint checkpoints/latest

# Run inference
python -m src.inference.generate \
  --checkpoint checkpoints/latest \
  --prompt "def fibonacci(n):"

# Start API
uvicorn backend.main:app --reload
```

Each advertised command must be functional with appropriate installed dependencies and required input files.

Frontend:

```bash
cd frontend
npm install
npm run dev
npm run build
```

Provide `.env.example` with clearly documented safe defaults.

---

# 21. SECURITY AND PRIVACY

Enforce:

- No credentials in source code
- No secrets in training records
- No unauthorized repository access
- No automatic execution of generated code on the host
- Sandboxed testing
- Safe ZIP extraction
- Restricted filesystem paths
- Bounded CPU, memory and execution time
- Network restrictions in code-execution sandboxes
- Secure storage configuration
- Proper input validation
- Audit logs for sensitive actions
- Safe API authentication for remote access

Public deployments must not expose unrestricted training-management or command-execution functions.

Document limitations of generated code and encourage review before deployment.

---

# 22. DEVELOPMENT IMPLEMENTATION ORDER

Build the project in this order.

## PHASE 1 — ENVIRONMENT

Implement:

- Repository
- Dependencies
- GPU detection
- CUDA verification
- Storage configuration
- Development scripts

Acceptance:

The environment report works and dependencies can be installed.

## PHASE 2 — CORE TRANSFORMER

Implement:

- RMSNorm
- RoPE
- Causal attention
- SwiGLU
- Transformer blocks
- Model configuration
- Forward pass
- Loss calculation
- Unit tests

Acceptance:

Both model configurations initialize, report their actual parameter counts, and pass shape/gradient tests.

## PHASE 3 — TOKENIZER

Implement original BPE tokenizer training.

Acceptance:

Saved tokenizer encodes and reconstructs representative code samples correctly.

## PHASE 4 — DATA ENGINE

Implement source registration, filtering, deduplication, splitting, tokenization, sharding and streaming.

Acceptance:

A small authorized sample dataset can be processed end to end.

## PHASE 5 — TRAINING ENGINE

Implement:

- Training loop
- CUDA mixed precision
- Optimizer
- Scheduler
- Validation
- Checkpointing
- Resume
- Metrics

Acceptance:

Tiny-batch overfitting works and interrupted training resumes correctly.

## PHASE 6 — SMALL MODEL

Train PRIYAN-CODE-23M on a controlled dataset.

Acceptance:

Training produces genuine loss metrics and valid model checkpoints.

## PHASE 7 — MAIN MODEL

Initialize PRIYAN-CODE-134M with random parameters.

Run configuration and memory validation.

Train a small number of real steps before attempting the full pretraining budget.

Acceptance:

Model forward/backward, checkpointing, evaluation and autoregressive inference work.

## PHASE 8 — INSTRUCTION TRAINING

Build the instruction dataset and supervised fine-tuning engine for the model we pretrained.

Acceptance:

Instruction-format training and held-out evaluation execute successfully.

## PHASE 9 — BACKEND

Implement the FastAPI server and model-management services.

Acceptance:

Documented endpoints pass integration tests.

## PHASE 10 — FRONTEND

Build all requested dashboard pages.

Acceptance:

Frontend builds successfully and displays actual backend states.

## PHASE 11 — CODING AGENT

Implement the sandboxed experimental repository tool layer.

Acceptance:

A controlled coding task can create a proposed patch and execute an approved test without unrestricted host access.

## PHASE 12 — FINAL VALIDATION

Execute:

- Formatting/linting
- Unit tests
- Integration tests
- Tokenizer tests
- Model tests
- Dataset tests
- API tests
- Frontend build
- Environment checks
- Small-model inference
- Checkpoint recovery

Report actual results.

Fix all discovered blocking implementation errors.

Do not mark untested capabilities as completed.

---

# 23. FINAL DELIVERABLES

Provide:

1. Complete source-code repository.
2. Functional model architecture.
3. Original tokenizer training system.
4. Legitimate coding dataset processing pipeline.
5. Small-model training configuration.
6. Main-model training configuration.
7. Working training engine.
8. Resumable checkpoint system.
9. Real inference engine.
10. Evaluation engine.
11. FastAPI backend.
12. React + TypeScript dashboard.
13. Experimental sandboxed coding agent.
14. Automated test suite.
15. Installation documentation.
16. Training and hardware documentation.
17. Dataset provenance documentation.
18. Security documentation.
19. Model card template and reproducibility manifest.
20. Packaged project source archive if filesystem access is available.

Do not fabricate trained model weights, dataset content, checkpoints or evaluation results.

If the full-scale model cannot be trained because of hardware limitations, deliver the tested source code and checkpoint-compatible training pipeline and clearly describe the remaining compute-dependent steps.

---

# 24. FINAL QUALITY STANDARDS

The result must satisfy these standards:

**Architecture:** Real, original decoder-only Transformer with random initialization.

**Training:** Genuine gradient-based next-token learning.

**Tokenizer:** Original, versioned BPE tokenizer.

**Dataset:** Real, authorized data with provenance and validation.

**Inference:** Actual predictions from loaded model weights.

**Frontend:** A functional professional dashboard, not a visual-only prototype.

**Backend:** Working validated APIs.

**Security:** No unsafe automatic code execution.

**Testing:** Real tests and explicit results.

**Reproducibility:** All critical configurations and dependencies documented.

**Scalability:** No artificial 50 GB storage restriction.

**Transparency:** Never misrepresent training completion or model performance.

---

# 25. EXECUTION INSTRUCTIONS

Begin implementing PRIYAN CODE AI immediately.

Do not respond with only a high-level plan, abstract explanation or code snippets.

Create the actual project files.

First inspect the environment and available repository files, then implement the core architecture, tokenizer, data pipeline and training engine in that order.

Use a small CPU/GPU smoke-test dataset to establish that the project works before executing resource-intensive operations.

After each milestone:

- Run applicable tests.
- Fix implementation errors.
- Record files created.
- Record commands executed.
- Record measured results.
- Record remaining limitations.

Continue through the remaining phases as the execution environment and available resources allow.

Prioritize a working, verified original Coding LLM training system over superficial frontend complexity.

The final output must be a real, reproducible software project capable of training **PRIYAN-CODE-134M from scratch**, with a scalable path toward future larger coding models.

**PROJECT: PRIYAN CODE AI**

**BUILD GOAL: ORIGINAL CODING LLM + TRAINING INFRASTRUCTURE + INFERENCE PLATFORM + PROFESSIONAL DEVELOPER WORKSPACE.**

**START IMPLEMENTATION.**