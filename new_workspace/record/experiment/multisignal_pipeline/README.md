# Multisignal Pipeline

This folder is organized so that code, documentation, and run artifacts are separated.

## Code

- `run.py`
  - CLI entrypoint for the default benchmark.
- `benchmark_main.py`
  - candidate construction, evaluation orchestration, and benchmark selection.
- `data.py`
  - signal loading and cycle-level sequence assembly.
- `detectors_ml.py`
  - ML / phase-classification detectors.
- `detectors_rule.py`
  - rule-based detectors.
- `menses.py`
  - countdown logic and benchmark-side menses evaluation.
- `report_utils.py`
  - reporting and ranking output helpers.
- `protocol.py`
  - public protocol re-export.

## Code Subdirectories

- `config/`
  - mainline and experimental protocol constants.
- `core/`
  - localizer and stabilization utilities used by the main pipeline.
- `experimental/`
  - retained ablations and experimental comparators that are not the core shipped path.

## Documentation

- `docs/benchmark_methods_and_results.md`
  - benchmark method catalog and recorded results.
- `docs/open_source_notes.md`
  - notes about retained comparators and non-default helpers.

## Logs

- `logs/`
  - saved benchmark outputs and validation logs.

## Notes

- The default benchmark path is still `python run.py`.
- Some helpers are retained for debugging or retrospective analysis even if they are
  not called by the default benchmark.
