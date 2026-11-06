"""CHISS oversampling for binary imbalanced data.

Each minority anchor makes a fixed number of sampling attempts using its
admissible radii. Candidates must pass both overlap and duplicate checks.
The accepted sample count can be lower than the requested budget.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

_EPS = 2.220446049250313e-16


def chiss_oversample(X, y, minority_label=None, K0=9, Khat0=7, a=10, b=0.01, n_maj_max=1,
                     tau_overlap=0.5, delta_dup=0.005, distance='l1', target='default',
                     random_state=None, return_stats=False):
    """Generate synthetic minority samples from scaled binary training data.

    Scale X using training data only. random_state accepts an integer,
    None, or a NumPy Generator.
    """
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float); y = np.asarray(y).ravel()
    if minority_label is None:
        v, c = np.unique(y, return_counts=True); minority_label = v[np.argmin(c)]
    metric = 'manhattan' if distance == 'l1' else 'euclidean'
    m = y == minority_label; Xmin, Xmaj = X[m], X[~m]
    n_min, n_maj, d = len(Xmin), len(Xmaj), X.shape[1]
    stats = dict(requested=0, accepted=0, K_D=0, n_ind=0, active_anchors=0, admissible_anchors=0,
                 rejected_overlap=0, rejected_duplicate=0)
    empty = np.zeros((0, d))
    if n_min < 2 or n_maj == 0 or n_min >= n_maj:
        return (empty, stats) if return_stats else empty

    G = n_min / n_maj                                                       # Eq. (2)
    if target == 'default':                                                  # Eq. (3)
        theta = 0.7 if G < 0.01 else (1.0 if G < 0.2 else max(0.9, G))
    elif target == 'full':
        theta = 1.0
    elif target == 'conservative':
        theta = 0.5 if G < 0.01 else (0.9 if G < 0.2 else max(0.85, G))
    else:
        raise ValueError(f'unknown target {target!r}')
    n_syn = max(0, int(round(theta * n_maj - n_min)))                        # Eq. (4)
    KD = min(max(int(np.ceil(K0 / (1 + np.exp(a * (G - b))))), 1), len(X) - 1)   # Eq. (5)
    stats.update(requested=n_syn, K_D=KD)
    if n_syn == 0:
        return (empty, stats) if return_stats else empty
    n_ind = int(np.ceil(n_syn / n_min)); stats['n_ind'] = n_ind               # Line 3

    D, I = NearestNeighbors(n_neighbors=KD + 1, metric=metric).fit(X).kneighbors(Xmin)
    minpos = np.flatnonzero(m)
    nn_min = NearestNeighbors(metric=metric).fit(Xmin)
    nn_maj = NearestNeighbors(metric=metric).fit(Xmaj)
    out = []; n_good = 0

    for i in range(n_min):                                                   # Line 5
        if n_good >= n_syn:
            break
        idx, dst = I[i], D[i]
        s = np.flatnonzero(idx == minpos[i]); s = s[0] if len(s) else KD
        keep = np.arange(KD + 1) != s
        idx, dst = idx[keep][:KD], dst[keep][:KD]
        lab = y[idx]
        dl = np.sum(lab == minority_label) / max(np.sum(lab != minority_label), _EPS)   # Eq. (6)
        KL = int(np.ceil(Khat0 / (1 + np.exp(-a * (dl - b)))))              # Eq. (7)
        KF = max(1, min(KD, KL))                                             # Line 8
        Zd, Zl = dst[:KF], lab[:KF]                                          # Line 9
        maj_in = np.cumsum(Zl != minority_label)                             # Lines 10-16
        Q = Zd[(maj_in <= n_maj_max) & (Zd > 0)]
        if len(Q) == 0:
            continue
        stats['admissible_anchors'] += 1; stats['active_anchors'] += 1
        for _ in range(n_ind):                                               # Lines 17-33
            if n_good >= n_syn:
                break
            rc = Q[rng.integers(len(Q))]                                     # sample radius with replacement
            if distance == 'l1':
                e = -np.log(rng.random(d)); v = np.where(rng.random(d) > 0.5, 1., -1.) * e / e.sum()
            else:
                g = rng.standard_normal(d); v = g / np.linalg.norm(g)
            xs = Xmin[i] + rc * rng.random() ** (1 / d) * v
            dmn = nn_min.kneighbors(xs[None], 1)[0][0, 0]
            dmj = nn_maj.kneighbors(xs[None], 1)[0][0, 0]
            if dmj / (dmn + _EPS) >= tau_overlap and min(dmn, dmj) >= delta_dup:
                out.append(xs); n_good += 1
            elif min(dmn, dmj) < delta_dup:
                stats['rejected_duplicate'] += 1
            else:
                stats['rejected_overlap'] += 1
    S = np.array(out) if out else empty
    stats['accepted'] = len(S)
    return (S, stats) if return_stats else S


class CHISS:
    """imbalanced-learn style wrapper:  X_res, y_res = CHISS(random_state=28).fit_resample(X, y)"""

    def __init__(self, K0=9, Khat0=7, a=10, b=0.01, n_maj_max=1, tau_overlap=0.5, delta_dup=0.005,
                 distance='l1', target='default', random_state=None):
        self.K0, self.Khat0, self.a, self.b = K0, Khat0, a, b
        self.n_maj_max, self.tau_overlap, self.delta_dup = n_maj_max, tau_overlap, delta_dup
        self.distance, self.target, self.random_state = distance, target, random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y).ravel()
        v, c = np.unique(y, return_counts=True); ml = v[np.argmin(c)]
        S, self.stats_ = chiss_oversample(X, y, ml, self.K0, self.Khat0, self.a, self.b, self.n_maj_max,
                                          self.tau_overlap, self.delta_dup, self.distance, self.target,
                                          self.random_state, return_stats=True)
        return np.vstack([X, S]), np.concatenate([y, np.full(len(S), ml, dtype=y.dtype)])
