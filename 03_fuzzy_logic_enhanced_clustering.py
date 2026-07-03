"""
Appendix D. Academic Visualization for Clustering Results
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FEATURES = ["knowledge_score", "engagement_score", "assignment_completion", "response_time", "error_rate", "motivation_level", "cognitive_score", "learning_speed", "mental_load", "risk_score"]

def set_plot_style():
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})

def plot_pca_dataset(df, output_dir):
    X = df[FEATURES].values
    y = df["cluster_label"].values
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111)
    for label in sorted(np.unique(y)):
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, alpha=0.48, label=f"Cluster {label}")
        cx, cy = np.mean(X_pca[mask, 0]), np.mean(X_pca[mask, 1])
        ax.scatter(cx, cy, marker="X", s=180, linewidths=1.2)
        ax.text(cx, cy, f"  C{label}", fontsize=11, weight="bold")
    ax.set_title("Latent Structure of Synthetic Student Profiles")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=True, title="Reference groups")
    fig.tight_layout()
    fig.savefig(output_dir / "academic_pca_student_profiles.png", bbox_inches="tight")
    plt.close(fig)

def plot_metric_matrix(results_df, output_dir, filename):
    heat_df = results_df.set_index("Model")[["Accuracy", "ARI", "NMI", "Silhouette", "DBI", "CHI", "Training Time"]].copy()
    heat_norm = heat_df.copy()
    heat_norm["DBI"] = 1 / heat_norm["DBI"]
    heat_norm["Training Time"] = 1 / heat_norm["Training Time"]
    heat_norm = pd.DataFrame(MinMaxScaler().fit_transform(heat_norm), columns=["Accuracy", "ARI", "NMI", "Silhouette", "DBI score", "CHI", "Speed"], index=heat_df.index)
    fig = plt.figure(figsize=(10.2, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(heat_norm.values, aspect="auto")
    ax.set_xticks(np.arange(len(heat_norm.columns)))
    ax.set_yticks(np.arange(len(heat_norm.index)))
    ax.set_xticklabels(heat_norm.columns, rotation=35, ha="right")
    ax.set_yticklabels(heat_norm.index)
    for i in range(heat_norm.shape[0]):
        for j in range(heat_norm.shape[1]):
            ax.text(j, i, f"{heat_norm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Algorithm-Metric Suitability Matrix")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized score")
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)

def plot_accuracy_time(results_df, output_dir):
    fig = plt.figure(figsize=(8.4, 5.8))
    ax = fig.add_subplot(111)
    bubble_size = 260 + 1200 * results_df["ARI"].values
    ax.scatter(results_df["Training Time"], results_df["Accuracy"], s=bubble_size, alpha=0.38, edgecolors="none")
    for _, row in results_df.iterrows():
        ax.text(row["Training Time"], row["Accuracy"], f" {row['Model']}", va="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("Training time, seconds (log scale)")
    ax.set_ylabel("Accuracy after label alignment")
    ax.set_title("Accuracy-Computation Trade-off Map")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_computation_tradeoff_map.png", bbox_inches="tight")
    plt.close(fig)

def plot_rank_flow(results_df, output_dir):
    rank_metrics = ["Accuracy", "ARI", "NMI", "Silhouette", "DBI", "CHI", "Training Time"]
    rank_df = results_df.set_index("Model")[rank_metrics].copy()
    for col in ["Accuracy", "ARI", "NMI", "Silhouette", "CHI"]:
        rank_df[col] = rank_df[col].rank(ascending=False, method="min")
    for col in ["DBI", "Training Time"]:
        rank_df[col] = rank_df[col].rank(ascending=True, method="min")
    fig = plt.figure(figsize=(10.2, 5.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(rank_metrics))
    for alg in rank_df.index:
        ax.plot(x, rank_df.loc[alg].values, marker="o", linewidth=1.8, label=alg)
    ax.set_xticks(x)
    ax.set_xticklabels(["Acc.", "ARI", "NMI", "Sil.", "DBI", "CHI", "Time"])
    ax.set_yticks([1, 2, 3, 4])
    ax.invert_yaxis()
    ax.set_ylabel("Rank position")
    ax.set_title("Rank Flow of Algorithms Across Evaluation Criteria")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(output_dir / "algorithm_rank_flow_chart.png", bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic_student_profiling_dataset.xlsx")
    parser.add_argument("--classical", default="classical_clustering_results.csv")
    parser.add_argument("--fuzzy", default="fuzzy_enhanced_clustering_results.csv")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    set_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(args.dataset, sheet_name="Dataset")
    classical_df = pd.read_csv(args.classical)
    fuzzy_df = pd.read_csv(args.fuzzy)
    plot_pca_dataset(df, output_dir)
    plot_metric_matrix(classical_df, output_dir, "algorithm_metric_suitability_matrix.png")
    plot_metric_matrix(fuzzy_df, output_dir, "fuzzy_algorithm_metric_matrix.png")
    plot_accuracy_time(classical_df, output_dir)
    plot_rank_flow(classical_df, output_dir)
    print(f"Figures saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
