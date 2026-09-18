# Human behavioral-value analysis

`scripts/run_human_behavioral_value.py` evaluates whether the rate-free maintenance-state deviation adds held-out information about trial correctness. It reuses the human maintenance admission path and holds out all trials from each participant in turn.

The primary model adds state to categorical corpus/load, log-transformed delay spike rate, and past-trial history. It is compared on identical test trials with the combined nuisance model and with simpler state-only, corpus/load, rate, and history models. Every scaler, category encoding, and logistic fit is learned from training participants only. Results report natural-log loss, probability-scale calibration, and participant-level bootstrap intervals for baseline-minus-combined-state log-loss differences. The combined-nuisance contrast is the direct test of incremental predictive value from state.

Rows without an outcome or a requested baseline label are excluded with an explicit eligibility reason. If no held-out fold supports a requested comparison, the analysis returns `blocked`. Estimates are predictive associations in released task and correctness units, not causal effects.
