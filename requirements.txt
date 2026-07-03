"""
Appendix A. Synthetic Student Profiling Dataset Generation

This script generates a synthetic student profiling dataset for an adaptive higher
education platform. The dataset includes 4600 student profiles, 4 learner groups,
and 10 input features.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_PER_CLUSTER = 1150

FEATURES = [
    "knowledge_score", "engagement_score", "assignment_completion",
    "response_time", "error_rate", "motivation_level", "cognitive_score",
    "learning_speed", "mental_load", "risk_score"
]

CLUSTER_RANGES = {
    1: {"knowledge_score": (80, 100), "engagement_score": (75, 100), "assignment_completion": (80, 100), "response_time": (10, 80), "error_rate": (0, 20), "motivation_level": (75, 100), "cognitive_score": (75, 100), "learning_speed": (75, 100), "mental_load": (0, 30), "risk_score": (0, 25)},
    2: {"knowledge_score": (55, 79), "engagement_score": (50, 75), "assignment_completion": (55, 80), "response_time": (60, 140), "error_rate": (15, 40), "motivation_level": (50, 75), "cognitive_score": (55, 80), "learning_speed": (50, 75), "mental_load": (30, 60), "risk_score": (25, 50)},
    3: {"knowledge_score": (25, 54), "engagement_score": (65, 95), "assignment_completion": (50, 80), "response_time": (90, 200), "error_rate": (35, 65), "motivation_level": (65, 95), "cognitive_score": (40, 70), "learning_speed": (35, 65), "mental_load": (40, 70), "risk_score": (40, 70)},
    4: {"knowledge_score": (20, 55), "engagement_score": (10, 50), "assignment_completion": (10, 55), "response_time": (150, 300), "error_rate": (50, 90), "motivation_level": (10, 50), "cognitive_score": (25, 60), "learning_speed": (10, 55), "mental_load": (70, 100), "risk_score": (70, 100)},
}

def truncated_normal(rng: np.random.Generator, low: float, high: float, n: int, std_ratio: float = 0.18) -> np.ndarray:
    mean = (low + high) / 2
    std = max((high - low) * std_ratio, 1e-6)
    values = rng.normal(mean, std, size=n)
    return np.clip(values, low, high)

def generate_cluster(rng: np.random.Generator, cluster_label: int, n: int = N_PER_CLUSTER) -> pd.DataFrame:
    data = {}
    for feature, (low, high) in CLUSTER_RANGES[cluster_label].items():
        data[feature] = truncated_normal(rng, low, high, n)
    cluster_df = pd.DataFrame(data)

    # Mild correlation effects make profiles less purely interval-based.
    if cluster_label == 1:
        cluster_df["risk_score"] = np.clip(cluster_df["risk_score"] - 0.08 * cluster_df["knowledge_score"], 0, 100)
    elif cluster_label == 4:
        cluster_df["risk_score"] = np.clip(cluster_df["risk_score"] + 0.12 * cluster_df["mental_load"], 0, 100)
    elif cluster_label == 3:
        cluster_df["assignment_completion"] = np.clip(cluster_df["assignment_completion"] + 0.08 * cluster_df["engagement_score"], 0, 100)

    cluster_df["cluster_label"] = cluster_label
    return cluster_df

def build_metadata() -> pd.DataFrame:
    rows = [
        ["student_id", "Identifier", "Integer", "1-4600", "Unique synthetic student identifier"],
        ["knowledge_score", "Academic", "Float", "0-100", "General academic knowledge level"],
        ["engagement_score", "Behavioral", "Float", "0-100", "Learning platform activity and participation"],
        ["assignment_completion", "Academic", "Float", "0-100", "Percentage of completed assignments"],
        ["response_time", "Behavioral", "Float", "10-300 seconds", "Average response time"],
        ["error_rate", "Academic", "Float", "0-100", "Percentage of incorrect answers"],
        ["motivation_level", "Motivational", "Float", "0-100", "Learning motivation indicator"],
        ["cognitive_score", "Cognitive", "Float", "0-100", "Cognitive performance indicator"],
        ["learning_speed", "Cognitive", "Float", "0-100", "Speed of mastering materials"],
        ["mental_load", "Mental-load indicator", "Float", "0-100", "Learning-related mental load"],
        ["risk_score", "Risk indicator", "Float", "0-100", "Academic risk indicator"],
        ["cluster_label", "Reference label", "Integer", "1-4", "Used only for external evaluation"],
    ]
    return pd.DataFrame(rows, columns=["Feature name", "Category", "Data type", "Value range", "Description"])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_student_profiling_dataset.xlsx")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_STATE)
    parts = [generate_cluster(rng, cluster_label=j) for j in range(1, 5)]
    dataset = pd.concat(parts, ignore_index=True)
    dataset.insert(0, "student_id", np.arange(1, len(dataset) + 1))

    for feature in FEATURES:
        dataset[feature] = dataset[feature].round(2)

    distribution = dataset["cluster_label"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Number of samples")
    feature_ranges = []
    for cluster, ranges in CLUSTER_RANGES.items():
        for feature, (low, high) in ranges.items():
            feature_ranges.append([cluster, feature, low, high])
    feature_ranges = pd.DataFrame(feature_ranges, columns=["Cluster", "Feature", "Minimum", "Maximum"])
    summary_stats = dataset.groupby("cluster_label")[FEATURES].agg(["mean", "std"]).round(3)

    output_path = Path(args.output)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        dataset.to_excel(writer, sheet_name="Dataset", index=False)
        build_metadata().to_excel(writer, sheet_name="Metadata", index=False)
        distribution.to_excel(writer, sheet_name="Cluster_Distribution", index=False)
        feature_ranges.to_excel(writer, sheet_name="Feature_Ranges", index=False)
        summary_stats.to_excel(writer, sheet_name="Summary_Stats")

    print(f"Dataset saved to: {output_path.resolve()}")
    print(f"Shape: {dataset.shape}")

if __name__ == "__main__":
    main()
