# CHISS — Clean Hyperspherical Inter-Sample Similarity oversampling

MATLAB and Python implementations of CHISS for binary imbalanced classification.

CHISS generates minority samples inside locally admissible hyperspheres. Neighbourhood
information determines the sampling radii; overlap and duplicate checks filter the
generated candidates.

The implementation derives its neighbourhood sizes from the imbalance ratios and uses
a severity-dependent synthetic-sample budget. Sampling uses L1 hyperspheres by default.
Each minority anchor makes `ceil(n_syn / n_min)` attempts, subject to the global budget;
rejected samples are not retried or reallocated. As a result, the accepted count can
be below the requested count.

Default parameters: `K0 = 9`, `Khat0 = 7`, `a = 10`, `b = 0.01`, `n_maj_max = 1`,
`tau_overlap = 0.5`, `delta_dup = 0.005`, L1 distance.

## Python

```python
from chiss import CHISS, chiss_oversample

# imbalanced-learn style
X_res, y_res = CHISS(random_state=28).fit_resample(X_train_scaled, y_train)

# synthetic samples only
S = chiss_oversample(X_train_scaled, y_train, minority_label=1, random_state=28)
```

```bash
pip install -r python/requirements.txt
python python/scripts/example.py
```

`python/scripts/run_chiss_cv.py` runs CHISS under stratified k-fold cross-validation, fitting the
scaler and applying oversampling within the training folds only.

## MATLAB

```matlab
rng(28);
[Xsyn, balanced, ysyn, stats] = CHISS(Xtrain_scaled, ytrain, minorityLabel);     % defaults
[Xsyn, ~, ~, stats] = CHISS(Xtrain_scaled, ytrain, minorityLabel, 'nMajMax', 2); % options
```

## Notes

Inputs should be scaled before use — for example min-max scaling fitted on the training data — as
the duplication threshold `delta_dup` is an absolute distance.

Both implementations return a `stats` structure recording the requested and accepted counts, the
derived neighbourhood size, the number of admissible anchors, and the number of candidates
rejected by each filter.

Baseline methods used for comparison in the accompanying work are not redistributed here; they are
available from their original sources.

## Citation

Citation details will be added once the accompanying paper is published.

## License

MIT — see [LICENSE](LICENSE).
