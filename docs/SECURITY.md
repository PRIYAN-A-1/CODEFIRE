# Security boundaries

Loopback-only research service; no public shell or arbitrary training config endpoint. Generation does not execute code. Checkpoints use trusted-local pickle state; checksums detect corruption, not authenticity. No model/service API secret is required; local dashboard authentication is optional through PRIYAN_API_KEY.

Source regex filters reject common secrets/private keys and email-containing files, but cannot guarantee complete secret/PII removal. Audit corpora before training. Do not include secrets in source_url or permission fields. Repository helper confines paths to a resolved root, refuses hidden paths, bounds reads and rejects secret-looking content. It creates only a proposed diff, never applies it.

Optional `src.agent.sandbox.test_python` invokes a pre-pulled Docker image with no network, read-only root, dropped capabilities, unprivileged UID, memory/CPU/PID/time limits and a read-only temporary work directory. It never falls back to the host interpreter. Docker is required and was not available/validated end-to-end here; only command construction and no-fallback behavior are tested. Containers are not a sufficient boundary for hostile multi-tenant workloads: use a dedicated VM/sandbox infrastructure for public untrusted execution. Generated source is not executed anywhere in the smoke workflow; Python syntax validation uses AST parsing only.

The experimental agent is read/propose/optional sandbox utilities, not an autonomous coding planner. It does not hide model capability limits or fabricate successful repairs.
