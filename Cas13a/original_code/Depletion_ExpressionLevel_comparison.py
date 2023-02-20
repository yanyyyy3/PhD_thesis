#!/home/yanying/anaconda3/envs/py37/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:42:57 2019

@author: yanying
"""
#%%
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from collections import defaultdict
from statistics import median
import matplotlib.gridspec as gridspec
import numpy as np

'''
Figure3C
'''
file="targeting-nt_QLFTest.csv"
df=pandas.read_csv(file,sep='\t')
rRNA='rrsH	rrlH	rrfH	rrfG	rrlG	rrsG	rrfF	rrfD	rrlD	rrsD	rrsC	rrlC	rrfC	rrsA	rrlA	rrfA	rrsB	rrlB	rrfB	rrsE	rrlE	rrfE'
rRNAs=rRNA.split()
pal = sns.color_palette('pastel')
for i in df.index:
    if df['guides'][i][:4] in rRNAs:
        df.at[i,'type']='rRNA'
    if df['type'][i]=='NC':
        df.at[i,'type']='NT'

sns.set_style('whitegrid')
sns.set_palette('pastel')
fig, axes = plt.subplots(4,sharex=True,figsize=(6,4))
sns.distplot(df[df['type']=='essential']['logFC'],ax=axes[0],kde=False,bins=70)
sns.distplot(df[df['type']=='non-essential']['logFC'],ax=axes[1],kde=False,bins=85)
sns.distplot(df[df['type']=='rRNA']['logFC'],ax=axes[2],kde=False,bins=28)
sns.distplot(df[df['type']=='NT']['logFC'],ax=axes[3],kde=False,bins=30)
axes[0].set_title('essential',pad=1,fontsize='small')
axes[1].set_title('non-essential',pad=1,fontsize='small')
axes[2].set_title('rRNA',pad=1,fontsize='small')
axes[3].set_title('NT',pad=1,fontsize='small')
for i in range(4):
    # axes[i].set_ylabel('')
    axes[i].set_xlabel('')
    axes[i].tick_params(labelsize=8,pad=-.5)
NC=df[df['type']=='NT']
plt.vlines(np.quantile(NC['logFC'],0.01), ymin=0, ymax=300, color='r',linestyles='-',alpha=0.8)
plt.subplots_adjust(hspace=0.3)
plt.xlabel("log2FC",fontsize='medium')
plt.xlim(min(df['logFC']),0.5)
# plt.savefig("Figure3C.svg",dpi=400)
plt.show()
plt.close()

counts="TPM_counts.csv"
logfc="gRNAs.csv"
counts=pandas.read_csv(counts,sep='\t',index_col=0)
counts=counts.dropna()
logfc=pandas.read_csv(logfc,sep='\t')
logfc=logfc.dropna(subset=['logFC'])

logFC=defaultdict(list)
plot=defaultdict(list)
rRNA='rrsH	rrlH	rrfH	rrfG	rrlG	rrsG	rrfF	rrfD	rrlD	rrsD	rrsC	rrlC	rrfC	rrsA	rrlA	rrfA	rrsB	rrlB	rrfB	rrsE	rrlE	rrfE'
rRNAs=rRNA.split()
for i in list(set(logfc['geneid'])):
    gene_df=logfc[logfc['geneid']==i]
    if i in counts.index:
        plot['median'].append(median(gene_df['logFC']))
        plot['gene'].append(i)
        plot['gene_name'].append(list(gene_df['gene_name'])[0])
        if  list(gene_df['gene_name'])[0] in rRNAs:
            plot['type'].append('rRNA')
        else:
            plot['type'].append(list(gene_df['type'])[0])
    
        plot['early'].append(np.log2((counts['NT1early'][i]+counts['NT2early'][i])*0.5+0.01))
        plot['late'].append(np.log2((counts['NT1late'][i]+counts['NT2late'][i])*0.5+0.01))
    else:
        pass
plot=pandas.DataFrame.from_dict(plot)
ess=plot[plot['type']=='essential']
noness=plot[plot['type']=='non-essential']
rRNA=plot[plot['type']=='rRNA']

'''
Figure3D and S3B
scatterplot between gene expression level and median logFC for each gene
'''
plot=plot[plot['type']!='rRNA']
sns.set_palette("pastel")
plt.figure() 
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
ax_main.scatter(noness['early'],noness['median'],label='non-essential (Rs: %s)'%round(spearmanr(noness['median'],noness['early'],nan_policy='omit')[0],2),edgecolors='white',alpha=0.5)
ax_main.scatter(ess['early'],ess['median'],label='essential (Rs: %s)'%round(spearmanr(ess['median'],ess['early'],nan_policy='omit')[0],2),edgecolors='white',alpha=0.5)
ax_main.set(xlabel='Gene Expression Level (log2 TPM)',ylabel="median log2FC of gRNAs\nfor each gene")
ax_main.legend(fontsize='small')
ax_xDist.hist(plot['early'],bins=70,align='mid',alpha=1)
ax_xDist.set(ylabel='count')
ax_xDist.xaxis.set_tick_params(labelsize=5)
ax_yDist.hist(plot['median'],bins=50,orientation='horizontal',align='mid',alpha=1)
ax_yDist.set(xlabel='count')
ax_yDist.yaxis.set_tick_params(labelsize=5)
r,_=spearmanr(plot['median'],plot['early'],nan_policy='omit')
plt.suptitle("Early (OD 0.5)\nSpearman correlation: %s"%round(r,4),fontsize='medium',color='black')
# plt.savefig("Figure3D.svg")
plt.show()
plt.close()

plt.figure() 
gs = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
ax_main.scatter(noness['late'],noness['median'],label='non-essential (Rs: %s)'%round(spearmanr(noness['median'],noness['late'],nan_policy='omit')[0],2),edgecolors='white',alpha=0.5)
ax_main.scatter(ess['late'],ess['median'],label='essential (Rs: %s)'%round(spearmanr(ess['median'],ess['late'],nan_policy='omit')[0],2),edgecolors='white',alpha=0.5)

ax_main.set(xlabel='Gene Expression Level (log2 TPM)',ylabel="median log2FC of gRNAs\nfor each gene")
ax_main.legend(fontsize='small')
ax_xDist.hist(plot['late'],bins=70,align='mid',alpha=1)
ax_xDist.set(ylabel='count')
ax_xDist.xaxis.set_tick_params(labelsize=5)
ax_yDist.hist(plot['median'],bins=50,orientation='horizontal',align='mid',alpha=1)
ax_yDist.set(xlabel='count')
ax_yDist.yaxis.set_tick_params(labelsize=5)
r,_=spearmanr(plot['median'],plot['late'],nan_policy='omit')
plt.suptitle("Late (OD 0.8)\nSpearman correlation: %s"%round(r,4),fontsize='medium',color='black')
# plt.savefig("FigureS3B.svg")
plt.show()
plt.close()

'''
FigureS3DE
compare the depletion scores between essential and non-essential genes after removing the effects of expression level 
'''
import random
eva=defaultdict(list)
bars=defaultdict(list)
for i in range(10):
    # split the non-essential genes based on the range of each quantile of expression level of essential genes
    quantile=noness[(noness['early']>=np.quantile(ess['early'],0+i*0.1))&(noness['early']<=np.quantile(ess['early'],0.1+i*0.1))]
    #repeat 100 times
    for j in range(100):
        #randomly select 30 genes
        non_genes=random.sample(list(quantile['gene_name']),30)
        selected=quantile[quantile['gene_name'].isin(non_genes)]
        mean=np.mean(selected['early'])
        median=np.median(selected['early'])
        ##save the scores for essential genes
        eva['value'].append(np.mean(selected['early']))
        eva['value'].append(np.median(selected['early']))
        eva['value'].append(np.mean(selected['median']))
        eva['value'].append(np.median(selected['median']))
        eva['method']+=['expression_mean','expression_median','depletion_mean','depletion_median']
        eva['quantile']+=[i+1]*4
    ##save the scores for essential genes
    selected=ess[(ess['early']>=np.quantile(ess['early'],0+i*0.1))&(ess['early']<=np.quantile(ess['early'],0.1+i*0.1))]
    bars['value'].append(np.mean(selected['early']))
    bars['value'].append(np.median(selected['early']))
    bars['value'].append(np.mean(selected['median']))
    bars['value'].append(np.median(selected['median']))
    bars['method']+=['expression_mean','expression_median','depletion_mean','depletion_median']
    bars['quantile']+=[i+1]*4
    
eva=pandas.DataFrame.from_dict(eva)
bars=pandas.DataFrame.from_dict(bars)
sns.set_style('white')
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
legends=[mpatches.Patch(color='orange',label='essential genes'), 
         Line2D([0],[0],marker='o',color='w',markerfacecolor='#3b5998',markersize=10,label='non-essential genes')]
ylabels={'expression_mean':'Mean expression level','expression_median':'Median expression level','depletion_mean':'Mean gene depletion','depletion_median':'Median gene depletion'}
#FigureS3D
for i in ['expression_mean','expression_median','depletion_mean','depletion_median']:
    bar=bars[bars['method']==i]
    e=eva[eva['method']==i]
    ax= sns.barplot(data=bar,x='quantile',y='value',color='orange')
    ax = sns.swarmplot(data=e,x='quantile',y='value',color='#3b5998',alpha=0.7,s=3)
    sns.boxplot(
            medianprops={'visible': True,'color': 'k', 'ls': '-', 'lw': 2},
            whiskerprops={'visible': False},
            x="quantile",
            y="value",
            data=e,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)
    plt.ylabel(ylabels[i])
    plt.xlabel("Quantile")
    plt.legend(handles=legends)
    # plt.savefig("/reviewer_comment/"+i+".svg")
    plt.show()
    plt.close()
#FigureS3E
for i in range(10):
    quantile=noness[(noness['early']>=np.quantile(ess['early'],0+i*0.1))&(noness['early']<=np.quantile(ess['early'],0.1+i*0.1))]
    selected=ess[(ess['early']>=np.quantile(ess['early'],0+i*0.1))&(ess['early']<=np.quantile(ess['early'],0.1+i*0.1))]
    plt.figure()
    sns.histplot(quantile['early'],label='non-essential',bins=max(int(quantile.shape[0]/10),20),color='#3b5998')
    sns.histplot(selected['early'],label='essential',bins=10,color='orange')
    plt.xlabel("expression level (OD 0.4)")
    plt.title("Quantile %s"%(i+1))
    plt.legend()
    # plt.savefig("/reviewer_comment/quantile"+str(i+1)+".svg")
    plt.show()
    plt.close()






