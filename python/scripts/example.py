"""Minimal example: oversample a training set with CHISS."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chiss import CHISS
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(2, 1, (20, 2))]); y = np.r_[np.zeros(200), np.ones(20)]
X = (X - X.min(0)) / (X.max(0) - X.min(0))          # min-max scaling (fit on training data only)
sampler = CHISS(random_state=28)
X_res, y_res = sampler.fit_resample(X, y)
print('before:', np.bincount(y.astype(int)), ' after:', np.bincount(y_res.astype(int)), ' stats:', sampler.stats_)
