"""Verify the generated workbook, documented metrics and experimental grids."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_generator import regenerate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

def table(path):
    return pd.read_csv(ROOT / path)

def verify():
    generated = regenerate(42)
    workbook = pd.read_excel(ROOT/'synthetic_students_dataset.xlsx',sheet_name='Dataset')
    pd.testing.assert_frame_equal(workbook[generated.columns], generated, check_dtype=False)
    assert workbook.groupby('cluster_label').size().tolist() == [1150]*4
    provenance = json.loads((RESULTS/'generator_audit.json').read_text())
    assert [provenance[k] for k in ('exact_feature_cells','compared_cells','exact_feature_rows')] == [22,46000,0]
    clean = table('results/clean_full_metrics.csv').set_index(['model','stage'])
    expected = {('DBSCAN','Classical'):3861, ('DBSCAN','Hybrid'):4592,
                ('Fuzzy rules','Baseline'):4582, ('Crisp rules','Baseline'):4527}
    for key, count in expected.items():
        assert np.isclose(clean.loc[key,'accuracy_all'],count/4600)
    assert len(clean)==15
    grid = table('results/dbscan_grid.csv').set_index(['eps','min_samples'])
    assert len(grid)==28 and grid.index.nunique()==28
    assert grid.loc[(0.6,10),'n_clusters']==4
    assert np.isclose(grid.loc[(1.25,12),'accuracy_all'],.75)
    noise = table('noise/results/noise_runs.csv')
    at_half = noise[noise.level.eq(.50)]
    dbscan = at_half.query("model == 'DBSCAN' and stage == 'hybrid_label_free'").sort_values('rep')
    rule = at_half.query("model == 'Rules only' and stage == 'rule_only'").sort_values('rep')
    assert len(dbscan)==20 and np.allclose(dbscan.accuracy_all, rule.accuracy_all)
    counts = table('cluster_count/results/cluster_count_runs.csv')
    old = table('legacy/cluster_count/cluster_count_runs.csv')
    def gmm(values):
        return values.query("model == 'GMM' and requested_k == 4 and stage == 'classical'")
    for runs, expected in [(counts,20),(old,13)]:
        results = gmm(runs)
        assert len(results)==20 and results.accuracy_all.eq(1).sum()==expected
    check = table('results/failure_diagnostics.csv')
    check = check.groupby(['dataset','condition']).mean(numeric_only=True)
    assert np.isclose(check.loc[('Legacy','Imbalance FCM'),'score'],.467877,atol=1e-6)
    assert np.isclose(check.loc[('Legacy','Diffuse GMM'),'score'],.825)
    assert np.isclose(check.loc[('Primary','Diffuse GMM'),'score'],.75)
    print('Verified seed-42 data, provenance, clean results, DBSCAN grid, noise controls and seed diagnostics.')

if __name__ == '__main__':
    verify()
