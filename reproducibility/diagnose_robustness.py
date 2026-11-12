"""Measure majority fragmentation and outsider-component capture on both workbooks."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from validate_reviewers import (FEATURES, LOWER, UPPER, basic_metrics, fuzzy_c_means,
                                models, outsider_metrics, outsiders)

ROOT = Path(__file__).resolve().parent

def diagnostics(path, name):
    frame = pd.read_excel(path, sheet_name='Dataset')
    x = frame[FEATURES].to_numpy(float)
    y = frame.cluster_label.to_numpy(int)
    scaler = StandardScaler().fit(x)
    rows = []
    for rep in range(10):
        rng = np.random.default_rng(19300 + rep)
        idx = np.concatenate([rng.choice(np.flatnonzero(y==g), n, replace=True)
                              for g,n in enumerate([690,3220,460,230],1)])
        idx = idx[rng.permutation(len(idx))]
        z = scaler.transform(np.clip(x[idx], LOWER, UPPER))
        pred = fuzzy_c_means(z).argmax(1)
        majority = pred[y[idx]==2]
        rows.append(dict(dataset=name, condition='Imbalance FCM', rep=rep,
                         score=basic_metrics(y[idx],pred)['ari'],
                         majority_outside_largest=1-np.bincount(majority).max()/len(majority)))
        rng = np.random.default_rng(11300 + rep)
        added = outsiders(rng, 460, 'Diffuse')
        raw = np.vstack([x,added])
        truth = np.r_[y,np.full(len(added),5)]
        order = rng.permutation(len(raw))
        pred = models()['GMM'].fit_predict(scaler.transform(raw[order]))
        outcome = outsider_metrics(truth[order],pred)
        rows.append(dict(dataset=name, condition='Diffuse GMM', rep=rep,
                         score=outcome['known_accuracy'],
                         best_novel_component_purity=outcome['best_novel_cluster_purity'],
                         best_novel_component_recall=outcome['best_novel_cluster_recall']))
    return rows

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        rows = diagnostics(ROOT/'synthetic_students_dataset.xlsx','Primary')
        rows += diagnostics(ROOT/'legacy'/'synthetic_students_dataset.xlsx','Legacy')
    pd.DataFrame(rows).to_csv(ROOT/'results'/'failure_diagnostics.csv',index=False)
