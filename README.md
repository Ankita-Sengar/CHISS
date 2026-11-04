# CHISS — Clean Hyperspherical Inter-Sample Similarity oversampling

Reference implementations (MATLAB and Python) of the CHISS oversampling algorithm for imbalanced
binary classification.

CHISS constructs anchor-centred hyperspheres around minority samples, admits only those whose
neighbourhood is sufficiently clean, and samples uniformly inside the admissible region, filtering
each candidate for overlap with the majority class and for duplication of existing samples.

- Severity-aware synthetic target `theta(Delta_G)` and resulting budget `n_syn`
- Global and local neighbourhood sizes `K_D`, `K_L`, `K_F`, derived from the global and local
  imbalance ratios
- Admissible anchor-centred hyperspheres: a radius is admissible when its ball contains at most
  `n_maj_max` majority samples
- Uniform sampling inside L1 hyperspheres, with overlap (`tau_overlap`) and duplication
  (`delta_dup`) checks applied per candidate
- One attempt per required sample (`n_ind = ceil(n_syn / |X_min|)` per anchor); an admissible
  anchor–radius pair remains available after use, and rejected candidates are not replaced
- Generation stops once the global budget `n_syn` is reached; there is no quota reallocation, so
  the number generated may fall short of the budget but never exceeds it

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
