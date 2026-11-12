"""Sensitivity to prescribed cluster count on the fixed four-scenario dataset.

No reference labels enter model fitting, semantic matching, or order selection.
Stochastic models use 20 initialization seeds. Ward and DBSCAN run once.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score, completeness_score
from threadpoolctl import threadpool_limits
import shared_methods as common

ROOT=Path(__file__).resolve().parent
COUNTS=list(range(2,9))

def metrics(y,labels,z):
    return {**common.evaluate(y,labels,z),
            'homogeneity':homogeneity_score(y,labels),
            'completeness':completeness_score(y,labels)}

def process(name,k,seed,model,z,y,fuzzy):
    start=time.perf_counter();labels=model.fit_predict(z);seconds=time.perf_counter()-start
    base={'model':name,'requested_k':k,'seed':seed,'fit_seconds':seconds}
    out=[{**base,'stage':'classical','fallback_ratio':0.,'semantic_accuracy':np.nan,
          'aic':model.aic(z) if name=='GMM' else np.nan,
          'bic':model.bic(z) if name=='GMM' else np.nan,**metrics(y,labels,z)}]
    memberships=common.scenario_memberships(model,z,labels,fuzzy)
    soft=memberships.argmax(axis=1)+1;soft[memberships.sum(axis=1)==0]=-1
    hybrid,_=common.released.hybrid_predict(memberships,fuzzy,alpha=.8)
    for stage,pred,fallback in [('soft_conversion_only',soft,0.),('hybrid_label_free',hybrid,float(np.mean(memberships.sum(axis=1)==0)))]:
        out.append({**base,'stage':stage,'fallback_ratio':fallback,
                    'semantic_accuracy':float(np.mean(pred==y)),'aic':np.nan,'bic':np.nan,**metrics(y,pred,z)})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeats',type=int,default=20);args=ap.parse_args()
    output=ROOT/'results';output.mkdir(exist_ok=True)
    data=ROOT.parent/'synthetic_students_dataset.xlsx';df=pd.read_excel(data,sheet_name='Dataset')
    z=StandardScaler().fit_transform(df[common.FEATURES]);y=df.cluster_label.to_numpy(int)
    fuzzy=common.released.compute_fuzzy_membership(df)
    seeds=list(range(42,42+args.repeats))
    protocol={
      'source_repository':'https://github.com/Khasanovpython/Adaptive-learning-Clustering-Research',
      'source_commit':'3fe7c251b9aa8e1d6d622ae16398ceb065a86921',
      'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
      'data':'Seed-42 documented-generator benchmark; 4600 rows, 10 features, four balanced generating scenarios',
      'experiment':'Vary requested number of model clusters; true scenario count remains four',
      'requested_k':COUNTS,'initialization_seeds':seeds,
      'repetitions':'K-means and GMM: 20 seeds; Ward: once per k; DBSCAN: once as count-free reference',
      'preprocessing':'StandardScaler fitted to the same ten clean features; ID and labels excluded',
      'K-means':dict(init='k-means++',n_init=30,max_iter=500,tol=1e-4),
      'Hierarchical':dict(linkage='ward',metric='euclidean'),
      'GMM':dict(covariance_type='full',init_params='kmeans',n_init=10,max_iter=500,reg_covar=1e-6),
      'DBSCAN':dict(eps=.6,min_samples=10,metric='euclidean',n_jobs=1),
      'fuzzy':'Released ten-feature mean-aggregation compatibility functions; fixed four scenario outputs',
      'semantic_mapping':'Same target-label-free one-to-one Hungarian matching as noise experiment. More than four detected clusters leave some clusters unmatched; these points use rule-only fallback.',
      'hybrid':dict(alpha=.8,distance_temperature=1.2),
      'primary_metrics':['ARI','homogeneity','completeness'],
      'accuracy':'One-to-one Hungarian-aligned accuracy; rejected and unmatched columns receive no credit. This score penalizes pure subclusters when k exceeds four.',
      'model_order_selection':'For each stochastic seed choose k by each internal index or AIC/BIC without reference labels. Ward has one deterministic selection.',
      'internal_indices':'Silhouette uses fixed 1200-observation sample, seed42; DBI/CHI use full assigned set; DBSCAN noise excluded',
      'scope':'One fixed synthetic dataset; initialization variability is not dataset or sampling variability; no test of a different true scenario count',
      'versions':dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    }
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with threadpool_limits(limits=1):
      for k in COUNTS:
        for seed in seeds:
          km=KMeans(n_clusters=k,init='k-means++',n_init=30,max_iter=500,tol=1e-4,random_state=seed)
          gm=GaussianMixture(n_components=k,covariance_type='full',init_params='kmeans',n_init=10,max_iter=500,reg_covar=1e-6,random_state=seed)
          rows.extend(process('K-means',k,seed,km,z,y,fuzzy))
          rows.extend(process('GMM',k,seed,gm,z,y,fuzzy))
        rows.extend(process('Hierarchical',k,-1,AgglomerativeClustering(n_clusters=k,linkage='ward',metric='euclidean'),z,y,fuzzy))
        pd.DataFrame(rows).to_csv(output/'cluster_count_runs.csv',index=False)
        print(f'k={k} completed; {len(seeds)} initialization seeds per stochastic model',flush=True)
      rows.extend(process('DBSCAN',-1,-1,DBSCAN(eps=.6,min_samples=10,n_jobs=1),z,y,fuzzy))
      rule_labels=fuzzy.argmax(axis=1)+1
      rows.append({'model':'Rules only','requested_k':-1,'seed':-1,'fit_seconds':0.,'stage':'rule_only','fallback_ratio':1.,'semantic_accuracy':float(np.mean(rule_labels==y)),'aic':np.nan,'bic':np.nan,**metrics(y,rule_labels,z)})
    runs=pd.DataFrame(rows);runs.to_csv(output/'cluster_count_runs.csv',index=False)
    measures=['accuracy_all','ari','nmi','homogeneity','completeness','coverage','n_clusters','noise_ratio','fallback_ratio','semantic_accuracy','silhouette_assigned','dbi_assigned','chi_assigned','aic','bic']
    summary=runs.groupby(['requested_k','model','stage'])[measures].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(output/'cluster_count_summary.csv',index=False)
    choices=[]
    classical=runs[(runs.stage=='classical')&(runs.requested_k>0)]
    for (model,seed),group in classical.groupby(['model','seed']):
      criteria={'silhouette_assigned':'max','dbi_assigned':'min','chi_assigned':'max'}
      if model=='GMM':criteria.update(aic='min',bic='min')
      for criterion,direction in criteria.items():
        index=group[criterion].idxmin() if direction=='min' else group[criterion].idxmax()
        r=group.loc[index]
        choices.append(dict(model=model,seed=seed,criterion=criterion,selected_k=int(r.requested_k),value=float(r[criterion])))
    pd.DataFrame(choices).to_csv(output/'internal_order_selections.csv',index=False)
    diagnostics=[]
    with threadpool_limits(limits=1):
      for seed in [42,44]:
        gm=GaussianMixture(n_components=4,covariance_type='full',init_params='kmeans',n_init=10,max_iter=500,reg_covar=1e-6,random_state=seed)
        labels=gm.fit_predict(z)
        diagnostics.append({'seed':seed,'converged':bool(gm.converged_),'iterations':int(gm.n_iter_),
                            'lower_bound':float(gm.lower_bound_),'bic':float(gm.bic(z)),
                            'component_sizes':json.dumps(np.bincount(labels,minlength=4).tolist()),
                            **metrics(y,labels,z)})
    pd.DataFrame(diagnostics).to_csv(output/'gmm_k4_diagnostic.csv',index=False)
    print('Completed cluster-count experiment.',flush=True)

if __name__=='__main__':main()
