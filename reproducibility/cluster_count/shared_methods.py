"""Shared scoring and target-label-free semantic matching from the noise experiment."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("released_fuzzy", ROOT/"released_fuzzy.py")
released=importlib.util.module_from_spec(spec)
spec.loader.exec_module(released)
FEATURES=released.FEATURES


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
