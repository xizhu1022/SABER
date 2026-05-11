"""SABER probe and shared probe utilities.

`saber_probe.py` carries the core SABER algorithm (joint PK + CK heads with
4-cell routing and abstention). `probe_utils.py` holds the generic helpers
(data loading, group-aware splits, head trainers, per-head metrics) shared
with `saber.metrics` and `saber.baselines.evaluate`.
"""
