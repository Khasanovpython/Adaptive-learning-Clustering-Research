from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from validate_reviewers import FEATURES

root=Path(__file__).resolve().parent
results=root/'results'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
df=pd.read_excel(root/'synthetic_students_dataset.xlsx',sheet_name='Dataset')
pca=PCA(n_components=2)
projection=pca.fit_transform(StandardScaler().fit_transform(df[FEATURES]))
fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained')
for group,color in enumerate(['#0072B2','#D55E00','#009E73','#CC79A7'],1):
    mask=df.cluster_label==group
    ax.scatter(projection[mask,0],projection[mask,1],s=4,alpha=.43,label=f'Scenario {group}',color=color,rasterized=True)
ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
       ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
ax.legend(frameon=False,ncol=2,fontsize=8)
fig.savefig(results/'Figure_1_PCA.png',dpi=600)
fig.savefig(results/'Figure_1_PCA.pdf')
plt.close(fig)
scores=pd.read_csv(results/'individual_memberships.csv')
fig,axes=plt.subplots(1,2,figsize=(6.2,3.1),layout='constrained')
for ax,name,label in zip(axes,['max_membership','margin'],['(a) Largest rule membership','(b) Membership gap']):
    by_group=[scores.loc[scores.scenario==g,name].to_numpy() for g in range(1,5)]
    ax.boxplot(by_group,tick_labels=['1','2','3','4'],showfliers=False)
    ax.set(xlabel='Generating scenario',ylabel='Membership',title=label,ylim=(0,1))
    ax.grid(axis='y',alpha=.2)
fig.savefig(results/'Figure_3_Membership.png',dpi=600)
fig.savefig(results/'Figure_3_Membership.pdf')
plt.close(fig)
out=pd.read_csv(results/'outsider_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(6.2,3.1),layout='constrained')
for mode,color in [('Diffuse','#0072B2'),('Discordant compact','#D55E00')]:
    for ax,metric,stage in zip(axes,['novel_reject','best_novel_cluster_f1'],['Classical','Classical']):
        d=out[(out['mode']==mode)&(out.model=='DBSCAN')&(out.stage==stage)].sort_values('fraction')
        ax.errorbar(d.fraction*100,d[metric+'_mean'],yerr=d[metric+'_std'],marker='o',label=mode,color=color,capsize=3)
for ax,title in zip(axes,['(a) Outside-profile rejection','(b) Best outside-cluster F1']):
    ax.set(xlabel='Added profiles (% of clean sample)',ylabel='Fraction',title=title,xticks=[5,10,20],ylim=(-.04,1.08))
    ax.grid(axis='y',alpha=.2)
axes[0].legend(frameon=False,fontsize=8)
fig.savefig(results/'Figure_2_Outside_profiles.png',dpi=600)
fig.savefig(results/'Figure_2_Outside_profiles.pdf')
plt.close(fig)
