# Evaluation scope

The shipped evaluator measures held-out fixture cross-entropy/perplexity, CPU token-forward throughput and Python AST validity for two original prompts. Test sequences are sampled with replacement, so throughput/evaluation is a fixture measurement, not comprehensive coverage or a public benchmark. No generated source is executed on the host.

Compile success, unit-test pass rate, pass@1 and bug-fix success remain null because no isolated functional benchmark was run. No EvalPlus or LiveCodeBench integration/results are claimed. Syntax success does not establish correctness. Read evaluations/latest.json and reports for actual outputs and failure examples.

For functional scoring, obtain licensed held-out tasks, reject contaminated source, generate one independent candidate per task, run with a separately validated sandbox, and record all failures/timeouts. Add test scaffolding only inside the isolated runner. Train/instruction/benchmark separation must remain auditable.
