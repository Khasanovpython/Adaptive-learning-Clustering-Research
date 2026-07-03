"""
Appendix B. Classical Clustering Experiment

This script trains K-means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture
Model on the synthetic student profiling dataset and evaluates them using
external and internal clustering metrics.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURES = ["knowledge_score", "engagement_score", "assignment_completion", "response_time", "error_rate", "motivation_level", "cognitive_score", "learning_speed", "mental_load", "risk_score"]

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    cost_matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / len(y_true)

def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray, X_scaled: np.ndarray, training_time: float) -> dict:
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic_student_profiling_dataset.xlsx")
    parser.add_argument("--output", default="classical_clustering_results.csv")
    args = parser.parse_args()

    df = pd.read_excel(args.dataset, sheet_name="Dataset")
    X = df[FEATURES].values
    y_true = df["cluster_label"].values
    X_scaled = StandardScaler().fit_transform(X)

    models = {
        "K-means": KMeans(n_clusters=4, init="k-means++", n_init=30, max_iter=500, tol=1e-4, random_state=42),
        "Hierarchical": AgglomerativeClustering(n_clusters=4, linkage="ward", metric="euclidean"),
        "DBSCAN": DBSCAN(eps=0.60, min_samples=10, metric="euclidean", n_jobs=-1),
        "GMM": GaussianMixture(n_components=4, covariance_type="full", init_params="kmeans", n_init=10, max_iter=500, reg_covar=1e-6, random_state=42),
    }

    results = []
    for name, model in models.items():
        start_time = time.perf_counter()
        labels = model.fit_predict(X_scaled)
        training_time = time.perf_counter() - start_time
        results.append({"Model": name, **evaluate_clustering(y_true, labels, X_scaled, training_time)})

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(results_df.round(4))
    print(f"Results saved to: {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
