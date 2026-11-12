"""Test additive Gaussian measurement noise.

Reference labels are used only in evaluation. Cluster-to-scenario matching for
the revised hybrid uses mean fuzzy compatibility, never reference labels.
See protocol.json for the complete design and provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("released_fuzzy", ROOT / "released_fuzzy.py")
released = importlib.util.module_from_spec(spec)
spec.loader.exec_module(released)
FEATURES = released.FEATURES
LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

def models():
    return {
        "K-means": KMeans(n_clusters=4, init="k-means++", n_init=30, max_iter=500, tol=1e-4, random_state=42),
        "Hierarchical": AgglomerativeClustering(n_clusters=4, linkage="ward", metric="euclidean"),
        "DBSCAN": DBSCAN(eps=0.60, min_samples=10, metric="euclidean", n_jobs=1),
        "GMM": GaussianMixture(n_components=4, covariance_type="full", init_params="kmeans", n_init=10, max_iter=500, reg_covar=1e-6, random_state=42),
    }

def evaluate(y, labels, z):
    """Reject -1 rather than treating it as a matchable learner group."""
    assigned = labels != -1
    groups = np.unique(labels[assigned])
    actual = np.unique(y)
    cost = np.array([[np.sum((y == t) & (labels == g)) for g in groups] for t in actual])
    correct = 0
    if len(groups):
        rows, cols = linear_sum_assignment(-cost)
        correct = int(cost[rows, cols].sum())
    out = dict(accuracy_all=correct / len(y), coverage=float(assigned.mean()),
               accuracy_assigned=correct / int(assigned.sum()) if assigned.any() else np.nan,
               ari=adjusted_rand_score(y, labels), nmi=normalized_mutual_info_score(y, labels),
               n_clusters=len(groups), noise_ratio=float((~assigned).mean()),
               silhouette_assigned=np.nan, dbi_assigned=np.nan, chi_assigned=np.nan)
    if 1 < len(groups) < assigned.sum():
        za, la = z[assigned], labels[assigned]
        out.update(silhouette_assigned=silhouette_score(za, la, sample_size=min(1200, len(la)), random_state=42),
                   dbi_assigned=davies_bouldin_score(za, la), chi_assigned=calinski_harabasz_score(za, la))
    return out

def scenario_memberships(model, z, labels, fuzzy):
    """One-to-one semantic matching maximizes average rule compatibility.

    Extra detected clusters are left unmapped. Missing cluster memberships
    and DBSCAN noise fall back to rule-only memberships, explicitly reported.
    """
    groups = np.unique(labels[labels != -1])
    u = np.zeros((len(labels), 4))
    if len(groups) == 0:
        return u
    compatibility = np.array([fuzzy[labels == g].mean(axis=0) for g in groups])
    rows, cols = linear_sum_assignment(-compatibility)
    mapping = dict(zip(rows, cols))
    if isinstance(model, GaussianMixture):
        prob = model.predict_proba(z)[:, groups]
    else:
        centers = np.array([z[labels == g].mean(axis=0) for g in groups])
        distances = cdist(z, centers)
        logits = -(distances ** 2) / (2 * 1.2 ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
    for row, col in mapping.items():
        u[:, col] += prob[:, row]
    unmapped = np.array([g for i, g in enumerate(groups) if i not in mapping])
    u[(labels == -1) | np.isin(labels, unmapped)] = 0
    sums = u.sum(axis=1, keepdims=True)
    return np.divide(u, sums, out=np.zeros_like(u), where=sums > 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--output", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datafile = ROOT.parent / "synthetic_students_dataset.xlsx"
    df = pd.read_excel(datafile, sheet_name="Dataset")
    raw = df[FEATURES].to_numpy(float)
    y = df.cluster_label.to_numpy(int)
    scaler = StandardScaler().fit(raw)
    low = np.zeros(10); high = np.full(10, 100.0)
    low[FEATURES.index("response_time")] = 10
    high[FEATURES.index("response_time")] = 300
    protocol = {
        "dataset_sha256": hashlib.sha256(datafile.read_bytes()).hexdigest(),
        "source_repository": "https://github.com/Khasanovpython/Adaptive-learning-Clustering-Research",
        "source_commit": "3fe7c251b9aa8e1d6d622ae16398ceb065a86921",
        "n_profiles": len(df), "features": FEATURES, "noise_levels": LEVELS,
        "repeats_nonzero": args.repeats, "noise_seeds": list(range(20261110, 20261110 + args.repeats)),
        "noise": "x_noisy_ij=clip(x_ij + level * clean_feature_population_sd_j * epsilon_ij, physical_min_j, physical_max_j), epsilon iid N(0,1)",
        "scaler": "StandardScaler fitted once on the clean benchmark; reused at all noise levels",
        "clipping": "Physical domains only; no clipping to generating scenario intervals",
        "pairing": "Same Gaussian epsilon matrix per seed at every level and for every method",
        "algorithm_seed": 42, "hyperparameters": {k: v.get_params() for k, v in models().items()},
        "controls": ["Classical hard partition", "Soft-conversion-only with the same semantic mapping", "Rules only", "0.8/0.2 hybrid"],
        "hybrid": {"alpha": 0.8, "temperature": 1.2,
                   "semantic_mapping": "Hungarian assignment maximizing mean fuzzy compatibility of each detected cluster; no y_true input",
                   "zero_or_unmapped_algorithm_membership": "use fuzzy membership alone",
                   "rules": "Released ten-feature mean-aggregation rule compatibility, unchanged"},
        "metrics": "Hungarian-aligned accuracy with noise always rejected; ARI/NMI include -1 as rejection category; internal metrics exclude noise; report coverage separately",
        "interpretation": "Conditional repeated measurement perturbations of one fixed synthetic dataset; not independent datasets or real-world validation",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    with threadpool_limits(limits=1):
        for level in LEVELS:
            for rep in range(1 if level == 0 else args.repeats):
                seed = 20261110 + rep
                eps = np.random.default_rng(seed).standard_normal(raw.shape)
                proposed = raw + level * scaler.scale_ * eps
                noisy = np.clip(proposed, low, high)
                z = scaler.transform(noisy)
                frame = pd.DataFrame(noisy, columns=FEATURES)
                fuzzy = released.compute_fuzzy_membership(frame)
                if np.any(fuzzy.sum(axis=1) == 0):
                    raise ValueError("Uncovered fuzzy profile; needs explicit rejection policy")
                common = dict(level=level, rep=rep, seed=seed,
                              clipped_fraction=float(np.mean(proposed != noisy)),
                              realized_standardized_noise_rms=float(np.sqrt(np.mean(((noisy-raw)/scaler.scale_)**2))))
                rule_labels = fuzzy.argmax(axis=1) + 1
                rows.append({**common, "model": "Rules only", "stage": "rule_only", "fit_seconds": 0.0,
                             "fallback_ratio": 1.0, "semantic_accuracy": float(np.mean(rule_labels==y)), **evaluate(y, rule_labels, z)})
                for name, model in models().items():
                    start = time.perf_counter(); labels = model.fit_predict(z); elapsed = time.perf_counter()-start
                    rows.append({**common, "model": name, "stage": "classical", "fit_seconds": elapsed,
                                 "fallback_ratio": 0.0, "semantic_accuracy": np.nan, **evaluate(y, labels, z)})
                    u = scenario_memberships(model, z, labels, fuzzy)
                    # Isolate the soft conversion from the subsequent fuzzy blend.
                    # The same semantic mapping is retained, so this is a control
                    # for blending, not a rule-independent unsupervised baseline.
                    soft_labels = u.argmax(axis=1) + 1
                    soft_labels[u.sum(axis=1) == 0] = -1
                    rows.append({**common, "model": name, "stage": "soft_conversion_only", "fit_seconds": elapsed,
                                 "fallback_ratio": 0.0, "semantic_accuracy": float(np.mean(soft_labels==y)),
                                 **evaluate(y, soft_labels, z)})
                    hybrid, _ = released.hybrid_predict(u, fuzzy, alpha=0.8)
                    rows.append({**common, "model": name, "stage": "hybrid_label_free", "fit_seconds": elapsed,
                                 "fallback_ratio": float(np.mean(u.sum(axis=1)==0)),
                                 "semantic_accuracy": float(np.mean(hybrid==y)), **evaluate(y, hybrid, z)})
                pd.DataFrame(rows).to_csv(args.output / "noise_runs.csv", index=False)
                print(f"level={level:.2f} repeat={rep+1}/{1 if level==0 else args.repeats} completed", flush=True)
    runs = pd.DataFrame(rows)
    measures = ["accuracy_all", "ari", "nmi", "coverage", "accuracy_assigned", "noise_ratio", "n_clusters", "silhouette_assigned", "dbi_assigned", "chi_assigned", "fallback_ratio", "semantic_accuracy", "clipped_fraction", "realized_standardized_noise_rms"]
    summary = runs.groupby(["level", "model", "stage"])[measures].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.output / "noise_summary.csv", index=False)
    print("All runs completed.", flush=True)

if __name__ == "__main__":
    main()
