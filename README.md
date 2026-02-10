# zeroqkernel-ids

Research scaffold for:

- proving theoretical separation of Hyper-Gaussian Quantum Kernel in zero-day intrusion detection
- implementing reproducible experiments from code to reports

## Scope

This repository currently contains framework code and project structure only.
Core algorithms are intentionally left as TODOs or abstract interfaces.

## Research Tracks

1. Formal problem definition (OOD / measure-theoretic zero-day setting)
2. Classical kernel baseline limits
3. Hyper-Gaussian quantum feature map definition
4. Orthogonality and concentration-based separation claims
5. Detection score construction and empirical validation

## Layout

- `configs/`: dataset/model/experiment/sweep configs
- `src/zeroqkernel/`: core library code
- `scripts/`: CLI wrappers for routine workflows
- `data/`: raw/processed/splits placeholders
- `results/`: run artifacts and aggregate summary

## Quickstart

```bash
python -m venv .venv
# activate .venv
pip install -e .
python scripts/run_experiment.py --config configs/experiment/zeroday_attack_holdout.yaml
```

## Notes

- `datasets/splits.py` defines zero-day split strategy interfaces.
- `models/quantum/feature_maps.py` exposes Hyper-Gaussian feature map parameters.
- `models/quantum/kernels.py` separates single kernel and gram matrix APIs.

##License
- The copyright of this code belongs to @UCS LAB.

