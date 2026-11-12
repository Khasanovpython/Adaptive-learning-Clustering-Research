"""Reproducible supplemental experiments for manuscript ASI-4641164."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, kmeans_plusplus
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
import shared_methods as common

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
FEATURES = common.FEATURES
LOWER = np.array([0,0,0,10,0,0,0,0,0,0])
UPPER = np.array([100,100,100,300,100,100,100,100,100,100])
# Approximate design intervals from Table 4, in the workbook feature order.
BOXES = np.array([
 [(80,100),(75,100),(80,100),(10,80),(0,20),(75,100),(75,100),(75,100),(0,30),(0,25)],
 [(55,79),(50,75),(55,80),(60,140),(15,40),(50,75),(55,80),(50,75),(30,60),(25,50)],
 [(25,54),(65,95),(50,80),(90,200),(35,65),(65,95),(40,70),(35,65),(40,70),(40,70)],
 [(20,55),(10,50),(10,55),(150,300),(50,90),(10,50),(25,60),(10,55),(70,100),(70,100)],
])
PARAMETERS = {
 'L': (0,0,35,55), 'M': (35,60,85), 'H': (65,80,100,100),
 'fast': (10,10,70,110), 'medium_time': (70,140,210), 'slow': (170,220,300,300),
 'low_error': (0,0,15,30), 'medium_error': (20,45,70), 'high_error': (55,75,100,100),
}
RULES = [
 ['H','H','H','fast','low_error','H','H','H','L','L'],
 ['M','M','M','medium_time','medium_error','M','M','M','M','M'],
 ['L','H','M','medium_time','medium_error','H','M','M','M','M'],
 ['L','L','L','slow','high_error','L','L','L','H','H'],
]
DOMAINS = {'score': ['L','M','H'], 'time': ['fast','medium_time','slow'],
           'error': ['low_error','medium_error','high_error']}

def models(k=4):
    return {
      'K-means': KMeans(n_clusters=k, n_init=30, max_iter=500, random_state=42),
      'Ward': AgglomerativeClustering(n_clusters=k, linkage='ward'),
      'DBSCAN': DBSCAN(eps=.60, min_samples=10, n_jobs=1),
      'GMM': GaussianMixture(n_components=k, covariance_type='full', n_init=10, max_iter=500,
                            reg_covar=1e-6, random_state=42),
    }

def fuzzy_scores(raw, shift=None, rules=RULES, hard=False):
    """Rule means computed before across-scenario normalization."""
    shift = np.zeros(10) if shift is None else shift
    scores = []
    for rule in rules:
        indicators = []
        for j, term in enumerate(rule):
            points = np.array(PARAMETERS[term], dtype=float)
            interior = (points > LOWER[j]) & (points < UPPER[j])
            points[interior] += shift[j] * (UPPER[j] - LOWER[j])
            if hard:
                # A crisp counterpart using the alpha-cut of the same membership.
                mf = common.released.trimf if len(points) == 3 else common.released.trapmf
                val = (mf(raw[:,j], *points) >= .5).astype(float)
            else:
                mf = common.released.trimf if len(points) == 3 else common.released.trapmf
                val = mf(raw[:,j], *points)
            indicators.append(val)
        scores.append(np.mean(indicators, axis=0))
    return np.stack(scores, axis=1)

def normalize(scores):
    totals = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, totals, out=np.full_like(scores, .25), where=totals>0)

def basic_metrics(y, pred):
    groups = np.unique(pred[pred != -1])
    true = np.unique(y)
    table = np.array([[np.sum((y == t)&(pred == g)) for g in groups] for t in true])
    pairs = linear_sum_assignment(-table) if len(groups) else ([], [])
    matched = table[pairs].sum() if len(groups) else 0
    return {'accuracy': matched / len(y), 'ari': adjusted_rand_score(y,pred),
            'nmi': normalized_mutual_info_score(y,pred), 'coverage': np.mean(pred!=-1),
            'groups': len(groups)}

def semantic_stages(model, z, labels, fuzzy):
    u = common.scenario_memberships(model, z, labels, fuzzy)
    soft = u.argmax(axis=1)+1
    soft[u.sum(axis=1)==0] = -1
    hybrid, _ = common.released.hybrid_predict(u, fuzzy, alpha=.8)
    return {'Classical':labels, 'Soft only':soft, 'Hybrid':hybrid}

def fuzzy_c_means(z, k=4):
    best = None
    for seed in range(42,47):
        centers, _ = kmeans_plusplus(z, k, random_state=seed)
        for iteration in range(300):
            distance = np.maximum(cdist(z, centers, 'sqeuclidean'), 1e-12)
            weights = 1/distance
            u = weights / weights.sum(axis=1, keepdims=True)
            u2 = u**2
            updated = u2.T @ z / u2.sum(axis=0)[:,None]
            if np.linalg.norm(updated-centers) < 1e-6:
                centers = updated
                break
            centers = updated
        distance = np.maximum(cdist(z, centers, 'sqeuclidean'), 1e-12)
        weights = 1/distance
        u = weights / weights.sum(axis=1, keepdims=True)
        objective = np.sum(u*u*distance)
        if best is None or objective < best[0]:
            best = (objective, u)
    return best[1]

def summarize(rows, groups, name):
    data = pd.DataFrame(rows)
    data.to_csv(OUT / f'{name}_runs.csv', index=False)
    values = [c for c in data.select_dtypes(include='number').columns if c not in groups+['rep']]
    summary = data.groupby(groups, dropna=False)[values].agg(['mean','std','count'])
    summary.columns = ['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(OUT / f'{name}_summary.csv', index=False)

def outside_boxes(raw):
    inside = ((raw[:,None,:]>=BOXES[None,:,:,0]) & (raw[:,None,:]<=BOXES[None,:,:,1])).all(axis=2)
    return ~inside.any(axis=1)

def outsiders(rng, n, mode):
    center = [88,32,38,46,14,35,86,83,80,62]
    spread = [5,7,9,11,5,8,5,6,7,8]
    candidates = (rng.normal(center,spread,(n*3,10)) if mode=='Discordant compact'
                  else rng.uniform(LOWER,UPPER,(n*3,10)))
    candidates = np.clip(candidates, LOWER, UPPER)
    accepted = candidates[outside_boxes(candidates)][:n]
    assert len(accepted)==n and outside_boxes(accepted).all()
    return accepted

def outsider_metrics(y, pred):
    novel = y==5
    known = ~novel
    table = basic_metrics(y[known], pred[known])
    rejects = pred == -1
    candidates = []
    for cluster in np.unique(pred[~rejects]):
        chosen = pred==cluster
        hits = np.sum(novel&chosen)
        candidates.append((2*hits/(novel.sum()+chosen.sum()),hits/chosen.sum(),hits/novel.sum()))
    f1,purity,capture = max(candidates, default=(0.,0.,0.))
    return {'known_accuracy':table['accuracy'],'known_ari':table['ari'],
            'known_reject':rejects[known].mean(),'novel_reject':rejects[novel].mean(),
            'best_novel_cluster_f1':f1,'best_novel_cluster_purity':purity,
            'best_novel_cluster_recall':capture,'groups':len(np.unique(pred[~rejects]))}

def outsider_study(x, y, scaler):
    rows = []
    for mode in ['Discordant compact','Diffuse']:
        for frac in [.05,.10,.20]:
            for rep in range(10):
                rng = np.random.default_rng(11300+rep)
                novel = outsiders(rng, round(len(x)*frac),mode)
                raw = np.vstack([x,novel])
                truth = np.r_[y,np.full(len(novel),5)]
                order = rng.permutation(len(raw))
                raw, truth = raw[order], truth[order]
                z = scaler.transform(raw)
                f = normalize(fuzzy_scores(raw))
                for name, model in models().items():
                    labels = model.fit_predict(z)
                    for stage, prediction in semantic_stages(model,z,labels,f).items():
                        rows.append(dict(mode=mode, fraction=frac, rep=rep, model=name,
                                         stage=stage, **outsider_metrics(truth,prediction)))
                rows.append(dict(mode=mode, fraction=frac, rep=rep, model='Rules',
                                 stage='Rule only',**outsider_metrics(truth,f.argmax(1)+1)))
            print('Outsiders',mode,frac,flush=True)
    summarize(rows,['mode','fraction','model','stage'],'outsider')

def distribution_study(x,y,scaler):
    rows = []
    scenarios = [('Clean',0),('Overlap',.25),('Overlap',.50),('Overlap',.75),
                 ('Imbalance',1),('Independent indicators',1),('Rank dependence',1)]
    for mode, level in scenarios:
        for rep in range(1 if mode=='Clean' else 10):
            rng = np.random.default_rng(19300+rep)
            raw, truth = x.copy(), y.copy()
            if mode=='Overlap':
                for group in range(1,5):
                    mask = y==group
                    raw[mask] += level*(x.mean(axis=0)-x[mask].mean(axis=0))
            elif mode=='Imbalance':
                idx = np.concatenate([rng.choice(np.flatnonzero(y==g),n,replace=True)
                                      for g,n in enumerate([690,3220,460,230],1)])
                raw,truth = x[idx],y[idx]
            elif mode in ['Independent indicators','Rank dependence']:
                for group in range(1,5):
                    mask = y==group
                    original = x[mask]
                    if mode=='Independent indicators':
                        raw[mask] = np.column_stack([rng.permutation(col) for col in original.T])
                    else:
                        latent = rng.normal(size=(len(original),1))
                        sign = np.array([1,1,1,-1,-1,1,1,1,-1,-1])
                        factors = np.sqrt(.65)*latent*sign + np.sqrt(.35)*rng.normal(size=original.shape)
                        transformed = np.empty_like(original)
                        for j in range(10):
                            transformed[np.argsort(factors[:,j]),j] = np.sort(original[:,j])
                        raw[mask] = transformed
            raw = np.clip(raw,LOWER,UPPER)
            order = np.arange(len(raw)) if mode=='Clean' else rng.permutation(len(raw))
            raw,truth = raw[order],truth[order]
            z = scaler.transform(raw)
            f = normalize(fuzzy_scores(raw))
            for name, model in models().items():
                labels = model.fit_predict(z)
                for stage,prediction in semantic_stages(model,z,labels,f).items():
                    rows.append(dict(mode=mode,level=level,rep=rep,model=name,stage=stage,
                                     **basic_metrics(truth,prediction)))
            for name,pred in [('Fuzzy rules',f.argmax(1)+1),('Crisp rules',fuzzy_scores(raw,hard=True).argmax(1)+1),
                              ('Fuzzy C-means',fuzzy_c_means(z).argmax(1))]:
                rows.append(dict(mode=mode,level=level,rep=rep,model=name,stage='Baseline',
                                 **basic_metrics(truth,pred)))
        print('Distribution',mode,level,flush=True)
    summarize(rows,['mode','level','model','stage'],'distribution')

def rule_study(x,y,scaler):
    z = scaler.transform(x)
    fitted = [(name, model, model.fit_predict(z)) for name,model in models().items()]
    rows = []
    for mode in ['Original','Membership shift','Rule change','Combined']:
        for rep in range(1 if mode=='Original' else 20):
            rng = np.random.default_rng(29300+rep)
            shift = rng.choice([-.05,.05],10) if mode in ['Membership shift','Combined'] else None
            rules = [list(rule) for rule in RULES]
            if mode in ['Rule change','Combined']:
                for rule in rules:
                    idx = int(rng.integers(10))
                    terms = DOMAINS['time' if idx==3 else 'error' if idx==4 else 'score']
                    at = terms.index(rule[idx])
                    rule[idx] = rng.choice([terms[i] for i in range(3) if abs(at-i)==1])
            f = normalize(fuzzy_scores(x,shift,rules))
            rules_pred = f.argmax(1)+1
            rows.append(dict(mode=mode,rep=rep,model='Rules',**basic_metrics(y,rules_pred),
                             direct_accuracy=(rules_pred==y).mean()))
            for name,model,labels in fitted:
                pred = semantic_stages(model,z,labels,f)['Hybrid']
                rows.append(dict(mode=mode,rep=rep,model=name,**basic_metrics(y,pred),
                                 direct_accuracy=(pred==y).mean()))
    summarize(rows,['mode','model'],'rule_sensitivity')

def audit(x,y,scaler):
    z=scaler.transform(x)
    rows=[]
    for eps in [.5,.6,.7,.8,.9,1.,1.25]:
        for minimum in [5,10,12,20]:
            labels=DBSCAN(eps=eps,min_samples=minimum,n_jobs=1).fit_predict(z)
            row=common.evaluate(y,labels,z)
            rows.append(dict(eps=eps,min_samples=minimum,**row))
    pd.DataFrame(rows).to_csv(OUT/'dbscan_grid.csv',index=False)
    fitted=models()
    f=normalize(fuzzy_scores(x))
    _, counts=np.unique(y, return_counts=True)
    pca=PCA().fit(z)
    summary={'observations':len(y),'counts':counts.tolist(),'features':FEATURES,
             'pca_variance':pca.explained_variance_ratio_[:2].tolist(),
             'scenario_bounds_violations':[],
             'mean_max_membership':float(f.max(1).mean()),
             'noise_audit':{},
             'centering':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()}
    margin = np.sort(f,axis=1)[:,-1]-np.sort(f,axis=1)[:,-2]
    pd.DataFrame({'scenario':y,'max_membership':f.max(1),'margin':margin,
                  **{f'membership_{i+1}':f[:,i] for i in range(4)}}).to_csv(OUT/'individual_memberships.csv',index=False)
    clean=[]
    for name,m in fitted.items():
        labels=m.fit_predict(z)
        for stage,prediction in semantic_stages(m,z,labels,f).items():
            clean.append(dict(model=name,stage=stage,**common.evaluate(y,prediction,z)))
        if name=='DBSCAN':
            reject=labels==-1
            hybrid=semantic_stages(m,z,labels,f)['Hybrid']
            summary['noise_audit']={'noise_count':int(reject.sum()),
                'rules_correct_on_noise':int((f[reject].argmax(1)+1==y[reject]).sum()),
                'hybrid_correct_on_noise':int((hybrid[reject]==y[reject]).sum()),
                'hybrid_correct_on_assigned':int((hybrid[~reject]==y[~reject]).sum())}
    for name,pred in [('Fuzzy rules',f.argmax(1)+1),('Crisp rules',fuzzy_scores(x,hard=True).argmax(1)+1),
                      ('Fuzzy C-means',fuzzy_c_means(z).argmax(1))]:
        clean.append(dict(model=name,stage='Baseline',**common.evaluate(y,pred,z)))
    pd.DataFrame(clean).to_csv(OUT/'clean_full_metrics.csv',index=False)
    for group in range(1,5):
        data=x[y==group]
        bounds=BOXES[group-1]
        stats=pd.DataFrame(data,columns=FEATURES)
        stats.corr().to_csv(OUT/f'correlation_scenario_{group}.csv')
        stats.describe().to_csv(OUT/f'descriptive_scenario_{group}.csv')
        violations=((data<bounds[:,0])|(data>bounds[:,1])).sum(axis=0)
        summary['scenario_bounds_violations'].append(violations.tolist())
    summary['response_time_scenario2_range']=[float(x[y==2,3].min()),float(x[y==2,3].max())]
    summary['response_time_scenario3_range']=[float(x[y==3,3].min()),float(x[y==3,3].max())]
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(RULES,index=[1,2,3,4],columns=FEATURES).to_csv(OUT/'exact_rules.csv',index_label='scenario')
    (OUT/'membership_parameters.json').write_text(json.dumps(PARAMETERS,indent=2))
    print('Audit and rule parameters complete',flush=True)

def main():
    frame=pd.read_excel(ROOT/'synthetic_students_dataset.xlsx',sheet_name='Dataset')
    x=frame[FEATURES].to_numpy(float)
    y=frame.cluster_label.to_numpy(int)
    scaler=StandardScaler().fit(x)
    assert len(x)==4600 and x.shape[1]==10 and np.bincount(y)[1:].tolist()==[1150]*4
    assert np.allclose(normalize(fuzzy_scores(x)),common.released.compute_fuzzy_membership(frame),atol=1e-10)
    with threadpool_limits(limits=1):
        audit(x,y,scaler)
        rule_study(x,y,scaler)
        outsider_study(x,y,scaler)
        distribution_study(x,y,scaler)
    print('Completed supplemental experiments',flush=True)

if __name__=='__main__':
    main()
