# Appendix Codes: Fuzzy Logic-Enhanced Student Clustering Study

This folder contains Python codes used for the synthetic student profiling dataset experiment, classical clustering, fuzzy logic-enhanced clustering, and visualization.

## Files

1. `01_generate_synthetic_student_dataset.py` — generates a 4600-row synthetic student profiling dataset with 4 clusters and 1150 samples per cluster.
2. `02_classical_clustering_experiment.py` — trains K-means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Model and evaluates them.
3. `03_fuzzy_logic_enhanced_clustering.py` — applies fuzzy membership functions and compares classical and fuzzy-enhanced clustering outputs.
4. `04_academic_visualization.py` — generates article-ready visualizations.
5. `requirements.txt` — required Python packages.

## Usage

```bash
pip install -r requirements.txt
python 01_generate_synthetic_student_dataset.py --output synthetic_student_profiling_dataset.xlsx
python 02_classical_clustering_experiment.py --dataset synthetic_student_profiling_dataset.xlsx
python 03_fuzzy_logic_enhanced_clustering.py --dataset synthetic_student_profiling_dataset.xlsx
python 04_academic_visualization.py --dataset synthetic_student_profiling_dataset.xlsx --classical classical_clustering_results.csv --fuzzy fuzzy_enhanced_clustering_results.csv
```

## Note

The dataset generation script creates a controlled synthetic dataset based on predefined statistical ranges. In the article, GAN/CGAN is described as the conceptual generative extension for producing more realistic synthetic samples under cluster conditions.
