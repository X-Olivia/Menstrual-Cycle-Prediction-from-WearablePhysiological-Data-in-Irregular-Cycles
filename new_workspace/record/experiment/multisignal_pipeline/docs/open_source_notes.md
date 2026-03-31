# Multisignal Pipeline Notes

This directory mixes the current benchmark path with several retained comparators.

Mainline benchmark path:
- `run.py`
- `benchmark_main.py`
- `detectors_ml.py`
- `detectors_rule.py`
- `core/localizer.py`
- `menses.py`

Current default reporting uses:
- day-by-day prefix evaluation
- post-trigger evaluation
- anchor-day reporting

Retained but non-default helpers:
- `menses.evaluate_per_cycle_menses_len_from_daily_det`
  - kept for cycle-level retrospective checks
  - not called by `run.py`

Experimental / comparator candidates that appear in fast mode:
- `PhaseCls-Temp+HR[Bayesian]`
- `PhaseCls-Temp+HR[BayesianPersonalized]`
- `PhaseCls-ENS-Temp+HR[Champion-BayesianPersonalized]`

Why these are retained:
- they are part of the current ablation / comparator story
- they help separate pure mainline performance from Bayesian / personalized variants
- removing them would make recent logs and rankings harder to interpret

Interpretation note:
- non-personalized leaders should be compared against other non-personalized methods
- personalized candidates should be interpreted as a separate comparator family
