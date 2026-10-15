# Model card

Name: PRIYAN-CODE-SMOKE (delivery fixture)
Architecture: original decoder Transformer, 115,008 parameters.
Initialization: random.
Training data: original CC0 fixture, 3 train code files; held-out repositories never used to train model or tokenizer.
Training: CPU, 60 steps, 3,840 processed token exposures.
Intended use: training-system validation and education.
Not intended use: reliable production code generation or autonomous code changes.
Limitations: severe undertraining, tiny corpus, failed syntax examples, no functional benchmark.
License: project MIT, example data CC0; future training data may add obligations.

The 23M and 134M presets are architecture configurations, not delivered pretrained coding models. Replace this card with measured data/provenance/quality when training those models.
