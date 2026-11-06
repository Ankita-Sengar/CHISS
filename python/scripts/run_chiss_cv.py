"""Stratified 10-fold evaluation of CHISS.

The min-max scaling parameters are estimated from each training fold.
Oversampling is performed on that fold before classifier fitting;
the test fold is used only for evaluation.

Example:
  python run_chiss_cv.py --data /path/to/csvs --seeds 28,42 --impl python
"""
import os, sys, argparse, time, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, average_precision_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chiss import chiss_oversample

DATASETS = ['car.csv', 'clean1.csv', 'bupa.data_5.csv', 'Ecoli0vs1_8.csv', 'diabetes.csv', 'iris0_5.csv', 'phoneme.csv',
            'Yeast1_9.csv', 'haberman_4.csv', 'Vechicle2_19.csv', 'Vechicle3_19.csv', 'Parkisnson_23.csv', 'transfusion_4.csv',
            'Vechicle0_19.csv', 'credit_loss.csv', 'mofn.csv', 'churn_prediction.csv', 'flare.csv', 'thyroid1_6.csv',
            'pagebinary.csv', 'coil2000.csv', 'hypothyroid.csv', 'Wine-red-4_12.csv', 'Wine-red-94_12.csv',
            'Wine-red-37_12.csv', 'Wine-red-86_12.csv', 'Wine-red-39_12.csv', 'dis.csv']


def load(path):
    """Last column = label; non-numeric feature columns are integer-encoded; missing values -> 0."""
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].apply(lambda c: pd.to_numeric(c, errors='coerce') if c.dtype == object else c)
    for c in X.columns:
        if X[c].isna().all():
            X[c] = pd.factorize(df[c])[0]
    return X.fillna(0).to_numpy(float), pd.factorize(df.iloc[:, -1])[0]


def classifiers(seed):
    return {'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=101, random_state=seed),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(5),
            'Neural Network': MLPClassifier((10,), activation='relu', solver='sgd', learning_rate_init=0.1,
                                            alpha=0.0, max_iter=500, random_state=seed)}


def metrics(yt, yp, score):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel(); e = 1e-16
    rec, spec, prec = tp / (tp + fn + e), tn / (tn + fp + e), tp / (tp + fp + e)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(Accuracy=(tp + tn) / len(yt), Precision=prec, Recall=rec, F1=2 * prec * rec / (prec + rec + e),
                GMean=np.sqrt(rec * spec), MCC=(tp * tn - fp * fn) / den if den > 0 else 0.0,
                PR_AUC=average_precision_score(yt, score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder with the csv files')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--seeds', default='28,42')
    ap.add_argument('--impl', default='python', choices=['python', 'matlab'])
    ap.add_argument('--matlab-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'matlab'))
    ap.add_argument('--n-maj-max', type=float, default=1)
    ap.add_argument('--out', default='chiss_cv_results.csv')
    A = ap.parse_args()
    eng = None
    if A.impl == 'matlab':
        import matlab.engine, matlab
        eng = matlab.engine.start_matlab(); eng.addpath(os.path.abspath(A.matlab_dir), nargout=0)
    rows = []
    for f in A.datasets.split(','):
        X, y = load(os.path.join(A.data, f))
        for seed in [int(s) for s in A.seeds.split(',')]:
            for fold, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y)):
                Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
                lo, hi = Xtr.min(0), Xtr.max(0); rg = np.where(hi - lo == 0, 1, hi - lo)
                Xtr, Xte = (Xtr - lo) / rg, (Xte - lo) / rg
                v, c = np.unique(ytr, return_counts=True); ml = v[np.argmin(c)]
                yb, yteb = (ytr == ml).astype(int), (yte == ml).astype(int)
                t0 = time.process_time()
                if eng is None:
                    S = chiss_oversample(Xtr, ytr, ml, n_maj_max=A.n_maj_max, random_state=seed * 1000 + fold)
                else:
                    eng.rng(float(seed * 1000 + fold), nargout=0)
                    out = eng.CHISS(matlab.double(Xtr.tolist()), matlab.double(ytr.astype(float).reshape(-1, 1).tolist()),
                                    float(ml), 'nMajMax', float(A.n_maj_max), nargout=4)
                    S = np.asarray(out[0], float).reshape(-1, X.shape[1]) if np.asarray(out[0]).size else np.zeros((0, X.shape[1]))
                rt = time.process_time() - t0
                Xr = np.vstack([Xtr, S]); yr = np.concatenate([yb, np.ones(len(S), int)])
                for cn, cl in classifiers(seed + fold).items():
                    cl.fit(Xr, yr)
                    rows.append(dict(Dataset=f, Seed=seed, Fold=fold, Classifier=cn, Impl=A.impl, N_Syn=len(S), Runtime=rt,
                                     **metrics(yteb, cl.predict(Xte), cl.predict_proba(Xte)[:, 1])))
        print('done', f, flush=True)
        pd.DataFrame(rows).to_csv(A.out, index=False)
    if eng is not None: eng.quit()
    r = pd.DataFrame(rows)
    print(r.groupby('Classifier')[['Accuracy', 'Recall', 'F1', 'GMean', 'MCC', 'PR_AUC']].mean().round(4))
    print('-> ', A.out)


if __name__ == '__main__':
    main()
