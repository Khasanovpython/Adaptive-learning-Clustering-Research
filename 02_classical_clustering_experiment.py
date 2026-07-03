"""
Appendix C. Fuzzy Logic-Enhanced Clustering

This script applies fuzzy membership functions to student profiles and combines
fuzzy membership with classical clustering outputs.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURES = ["knowledge_score", "engagement_score", "assignment_completion", "response_time", "error_rate", "motivation_level", "cognitive_score", "learning_speed", "mental_load", "risk_score"]

def clustering_accuracy(y_true, y_pred):
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    cost_matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / len(y_true)

def evaluate_clustering(y_true, y_pred, X_scaled, training_time):
    unique_labels = np.unique(y_pred)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_ratio = np.mean(y_pred == -1) if -1 in unique_labels else 0.0
    if len(unique_labels) > 1 and len(unique_labels) < len(y_pred):
        silhouette = silhouette_score(X_scaled, y_pred, sample_size=min(1200, len(y_pred)), random_state=42)
        dbi = davies_bouldin_score(X_scaled, y_pred)
        chi = calinski_harabasz_score(X_scaled, y_pred)
    else:
        silhouette = np.nan
        dbi = np.nan
        chi = np.nan
    return {"Accuracy": clustering_accuracy(y_true, y_pred), "ARI": adjusted_rand_score(y_true, y_pred), "NMI": normalized_mutual_info_score(y_true, y_pred), "Silhouette": silhouette, "DBI": dbi, "CHI": chi, "Training Time": training_time, "Clusters": n_clusters, "Noise Ratio": noise_ratio}

def label_to_true_mapping(y_true, y_pred):
    pred_labels = [p for p in np.unique(y_pred) if p != -1]
    true_labels = list(np.unique(y_true))
    cost_matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return {pred_labels[j]: true_labels[i] for i, j in zip(row_ind, col_ind)}

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    idx = (x > a) & (x < b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)
    idx = (x >= b) & (x <= c)
    y[idx] = 1.0
    idx = (x > c) & (x < d)
    if d != c:
        y[idx] = (d - x[idx]) / (d - c)
    return np.clip(y, 0, 1)

def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    idx = (x > a) & (x <= b)
    if b != a:
        y[idx] = (x[idx] - a) / (b - a)
    idx = (x > b) & (x < c)
    if c != b:
        y[idx] = (c - x[idx]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0, 1)

def score_low(x): return trapmf(x, 0, 0, 35, 55)
def score_med(x): return trimf(x, 35, 60, 85)
def score_high(x): return trapmf(x, 65, 80, 100, 100)
def time_fast(x): return trapmf(x, 10, 10, 70, 110)
def time_med(x): return trimf(x, 70, 140, 210)
def time_slow(x): return trapmf(x, 170, 220, 300, 300)
def err_low(x): return trapmf(x, 0, 0, 15, 30)
def err_med(x): return trimf(x, 20, 45, 70)
def err_high(x): return trapmf(x, 55, 75, 100, 100)

def compute_fuzzy_membership(df):
    arr = {f: df[f].values for f in FEATURES}
    g1 = np.mean(np.vstack([score_high(arr["knowledge_score"]), score_high(arr["engagement_score"]), score_high(arr["assignment_completion"]), time_fast(arr["response_time"]), err_low(arr["error_rate"]), score_high(arr["motivation_level"]), score_high(arr["cognitive_score"]), score_high(arr["learning_speed"]), score_low(arr["mental_load"]), score_low(arr["risk_score"])]), axis=0)
    g2 = np.mean(np.vstack([score_med(arr["knowledge_score"]), score_med(arr["engagement_score"]), score_med(arr["assignment_completion"]), time_med(arr["response_time"]), err_med(arr["error_rate"]), score_med(arr["motivation_level"]), score_med(arr["cognitive_score"]), score_med(arr["learning_speed"]), score_med(arr["mental_load"]), score_med(arr["risk_score"])]), axis=0)
    g3 = np.mean(np.vstack([score_low(arr["knowledge_score"]), score_high(arr["engagement_score"]), score_med(arr["assignment_completion"]), time_med(arr["response_time"]), err_med(arr["error_rate"]), score_high(arr["motivation_level"]), score_med(arr["cognitive_score"]), score_med(arr["learning_speed"]), score_med(arr["mental_load"]), score_med(arr["risk_score"])]), axis=0)
    g4 = np.mean(np.vstack([score_low(arr["knowledge_score"]), score_low(arr["engagement_score"]), score_low(arr["assignment_completion"]), time_slow(arr["response_time"]), err_high(arr["error_rate"]), score_low(arr["motivation_level"]), score_low(arr["cognitive_score"]), score_low(arr["learning_speed"]), score_high(arr["mental_load"]), score_high(arr["risk_score"])]), axis=0)
    beta = np.vstack([g1, g2, g3, g4]).T
    return beta / (beta.sum(axis=1, keepdims=True) + 1e-12)

def distance_soft_membership(X_scaled, labels, y_true, temperature=1.2):
    pred_clusters = [p for p in np.unique(labels) if p != -1]
    centroids = np.vstack([X_scaled[labels == p].mean(axis=0) for p in pred_clusters])
    distances = cdist(X_scaled, centroids, metric="euclidean")
    similarity = np.exp(-(distances ** 2) / (2 * temperature ** 2))
    prob = similarity / (similarity.sum(axis=1, keepdims=True) + 1e-12)
    mapping = label_to_true_mapping(y_true, labels)
    U = np.zeros((len(labels), 4))
    for idx, pred_cluster in enumerate(pred_clusters):
        if pred_cluster in mapping:
            U[:, mapping[pred_cluster] - 1] += prob[:, idx]
    U[labels == -1, :] = 0.0
    row_sum = U.sum(axis=1, keepdims=True)
    nonzero = row_sum[:, 0] > 0
    U[nonzero] = U[nonzero] / row_sum[nonzero]
    return U

def gmm_membership_aligned(gmm_model, X_scaled, labels, y_true):
    prob = gmm_model.predict_proba(X_scaled)
    mapping = label_to_true_mapping(y_true, labels)
    U = np.zeros((len(labels), 4))
    for component in range(prob.shape[1]):
        if component in mapping:
            U[:, mapping[component] - 1] += prob[:, component]
    return U / (U.sum(axis=1, keepdims=True) + 1e-12)

def hybrid_predict(U_algorithm, U_fuzzy, alpha=0.80):
    U = alpha * U_algorithm + (1 - alpha) * U_fuzzy
    zero_algorithm_membership = U_algorithm.sum(axis=1) == 0
    U[zero_algorithm_membership] = U_fuzzy[zero_algorithm_membership]
    U = U / (U.sum(axis=1, keepdims=True) + 1e-12)
    return np.argmax(U, axis=1) + 1, U

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic_student_profiling_dataset.xlsx")
    parser.add_argument("--output", default="fuzzy_enhanced_clustering_results.csv")
    parser.add_argument("--alpha", default=0.80, type=float)
    args = parser.parse_args()

    df = pd.read_excel(args.dataset, sheet_name="Dataset")
    X = df[FEATURES].values
    y_true = df["cluster_label"].values
    X_scaled = StandardScaler().fit_transform(X)

    classical_labels = {}
    classical_times = {}
    start = time.perf_counter(); kmeans = KMeans(n_clusters=4, init="k-means++", n_init=30, max_iter=500, tol=1e-4, random_state=42); classical_labels["K-means"] = kmeans.fit_predict(X_scaled); classical_times["K-means"] = time.perf_counter() - start
    start = time.perf_counter(); hc = AgglomerativeClustering(n_clusters=4, linkage="ward", metric="euclidean"); classical_labels["Hierarchical"] = hc.fit_predict(X_scaled); classical_times["Hierarchical"] = time.perf_counter() - start
    start = time.perf_counter(); db = DBSCAN(eps=0.60, min_samples=10, metric="euclidean", n_jobs=-1); classical_labels["DBSCAN"] = db.fit_predict(X_scaled); classical_times["DBSCAN"] = time.perf_counter() - start
    start = time.perf_counter(); gmm = GaussianMixture(n_components=4, covariance_type="full", init_params="kmeans", n_init=10, max_iter=500, reg_covar=1e-6, random_state=42); classical_labels["GMM"] = gmm.fit_predict(X_scaled); classical_times["GMM"] = time.perf_counter() - start

    start = time.perf_counter()
    U_fuzzy = compute_fuzzy_membership(df)
    fuzzy_time = time.perf_counter() - start

    U_algorithm = {
        "K-means": distance_soft_membership(X_scaled, classical_labels["K-means"], y_true),
        "Hierarchical": distance_soft_membership(X_scaled, classical_labels["Hierarchical"], y_true),
        "DBSCAN": distance_soft_membership(X_scaled, classical_labels["DBSCAN"], y_true),
        "GMM": gmm_membership_aligned(gmm, X_scaled, classical_labels["GMM"], y_true),
    }

    results = []
    for algorithm in ["K-means", "Hierarchical", "DBSCAN", "GMM"]:
        hybrid_labels, _ = hybrid_predict(U_algorithm[algorithm], U_fuzzy, alpha=args.alpha)
        metrics = evaluate_clustering(y_true, hybrid_labels, X_scaled, classical_times[algorithm] + fuzzy_time)
        results.append({"Model": f"{algorithm} + Fuzzy", **metrics})

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(results_df.round(4))
    print(f"Fuzzy-enhanced results saved to: {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
