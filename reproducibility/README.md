# ASI-4641164 reproducibility archive

The primary workbook has 4,600 synthetic profiles and was generated with the documented Appendix A implementation at seed 42. The legacy workbook and its prior comparison outputs are in `legacy/`. The two workbooks are distinct. The legacy file is archived for comparison because the published generator does not reproduce its rows. Both workbooks are synthetic.

The original release is [Khasanovpython/Adaptive-learning-Clustering-Research](https://github.com/Khasanovpython/Adaptive-learning-Clustering-Research/tree/3fe7c251b9aa8e1d6d622ae16398ceb065a86921). At that commit, `requirements.txt` contains the Python generator. The archive keeps its implementation as `repository_generator.py` and supplies an actual dependency file. The revision code and results are provided in this archive.

## Reproduce the primary experiments

Use Python 3.12.14 and install the pinned requirements with `python -m pip install -r requirements.txt`. Run the following from the archive root.

```sh
python repository_generator.py --output synthetic_students_dataset.xlsx
python validate_reviewers.py
python noise/run_noise_experiment.py
python cluster_count/run_cluster_count_experiment.py
python audit_generator.py
python diagnose_robustness.py
python make_figures.py
python make_workflows.py
python verify_primary.py
```

The workbook generator writes two-decimal values. ZIP metadata inside a regenerated workbook may change its byte checksum. The verifier compares the actual sheet values against the fixed seed. Result files that report runtime or workbook byte hashes can differ across machines or regenerations. The benchmark scripts limit numerical thread pools for fitting.

## Manuscript outputs

| Manuscript location | Reproduction source |
|---|---|
| Tables 1–2 | `results/membership_parameters.json`, `results/exact_rules.csv` |
| Tables 3–5 | Workbook metadata, scenario counts and initial generator intervals |
| Tables 7–8 | `results/clean_full_metrics.csv` |
| Table 9 | `results/distribution_summary.csv` |
| Table 10 | `cluster_count/results/cluster_count_summary.csv` |
| Table 11 | `results/outsider_summary.csv` |
| Table 12 | `results/dbscan_grid.csv`, complete 28-setting grid |
| Figures 1, 3 | `make_workflows.py` |
| Figures 4–6 | `make_figures.py` |
| Section 4.3.1 | `results/rule_sensitivity_summary.csv` |
| Sections 3.14, 4.4 | `results/generator_audit.json`, `results/regenerated_summary.csv`, `results/failure_diagnostics.csv`, and the noise/count subdirectories |

The rule-based semantic alignment uses mean rule compatibility. Generating labels are used for scoring and to construct controlled perturbations. The scenario ranges in the workbook are initial generation intervals. After the documented dependence adjustment, 75 scenario-3 completion values exceed their initial interval but remain within their physical domain.

To rerun the legacy analyses, work in a separate copy of the archive, place `legacy/synthetic_students_dataset.xlsx` at the root under `synthetic_students_dataset.xlsx`, and run the relevant benchmark scripts. The primary verification checks are specific to the generated workbook.
