"""Compare the workbook to the public Appendix A generator."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import repository_generator as generator
from validate_reviewers import models, normalize, fuzzy_scores, semantic_stages, basic_metrics, summarize, BOXES

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
def regenerate(seed):
    rng=np.random.default_rng(seed)
    frame=pd.concat([generator.generate_cluster(rng,g) for g in range(1,5)],ignore_index=True)
    frame[generator.FEATURES]=frame[generator.FEATURES].round(2)
    return frame

def main():
    original=pd.read_excel(ROOT/'legacy'/'synthetic_students_dataset.xlsx',sheet_name='Dataset')
    target=original[generator.FEATURES].to_numpy()
    first=regenerate(generator.RANDOM_STATE)
    matches=np.isclose(target,first[generator.FEATURES].to_numpy(),rtol=0,atol=1e-10)
    audit={'repository_seed':generator.RANDOM_STATE,'compared_cells':int(matches.size),
           'exact_feature_cells':int(matches.sum()),'exact_feature_rows':int(matches.all(axis=1).sum()),
           'source':'https://github.com/Khasanovpython/Adaptive-learning-Clustering-Research/blob/3fe7c251b9aa8e1d6d622ae16398ceb065a86921/requirements.txt',
           'model':'Independent Gaussian draws within scenario and feature, clipped to interval; sigma=0.18 interval width. Three additive adjustments in Appendix A alter selected dependencies; those features are then clipped to [0,100].',
           'scope':'Appendix A generates the primary benchmark at seed 42. The process that created the separately archived legacy workbook is unconfirmed.'}
    (OUT/'generator_audit.json').write_text(json.dumps(audit,indent=2))
    rows=[]
    with threadpool_limits(limits=1):
        for rep,seed in enumerate(range(42,52)):
            frame=regenerate(seed)
            x=frame[generator.FEATURES].to_numpy()
            y=frame.cluster_label.to_numpy()
            b=BOXES[y-1]
            violations=np.sum((x<b[:,:,0]) | (x>b[:,:,1]))
            z=StandardScaler().fit_transform(x)
            f=normalize(fuzzy_scores(x))
            for name,m in models().items():
                labels=m.fit_predict(z)
                for stage,pred in semantic_stages(m,z,labels,f).items():
                    rows.append(dict(model=name,stage=stage,rep=rep,seed=seed,bounds_violation_cells=violations,
                                     **basic_metrics(y,pred)))
            rows.append(dict(model='Fuzzy rules',stage='Rule only',rep=rep,seed=seed,bounds_violation_cells=violations,
                             **basic_metrics(y,f.argmax(1)+1)))
    summarize(rows,['model','stage'],'regenerated')
    print(audit)

if __name__ == '__main__':
    main()
