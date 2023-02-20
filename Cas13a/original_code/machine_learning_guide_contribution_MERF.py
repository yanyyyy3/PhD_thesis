#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:35:49 2020

@author: yanying
"""
#%%
import pandas
from merf import MERF
import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import spearmanr,pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from collections import defaultdict
import os
import logging
import time
import statistics
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['figure.dpi'] = 300
import argparse
import sys
start=time.time()
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to separate gene and guide effects using MERF (tested version 1.0).

Example: python machine_learning_guide_contribution_MERF.py
                  """)
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.2")
args = parser.parse_args()
folds=args.folds
test_size=args.test_size
output_file_name=args.output
def self_encode(sequence):
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','U','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded
def DataFrame_input(df):
    logging_file= open(output_file_name + '/log.txt','a')
    df=df.replace([np.inf, -np.inf], np.nan).dropna(subset=['logFC','expression_level_early'])
    df=df[~(df['PFS']=='A')]
    logging_file.write("Number of guides: %s \n" % df.shape[0])
    guide_sequence_set=list(dict.fromkeys(df['seq']))
    for i in list(set(df['geneid'])):
        gene_df=df[df['geneid']==i]
        median=statistics.median(gene_df['logFC'])
        for j in gene_df.index:
            df.at[j,'median']=median
    ### one
    PFS_encoded=[]
    sequence_encoded=[]

    for i in df.index:
        df.at[i,'guideid']=guide_sequence_set.index(df['seq'][i])
        PFS_encoded.append(self_encode(df['PFS'][i]))
        sequence_encoded.append(self_encode(df['seq'][i].replace("T","U")))
        df.at[i,'geneid']=int(df['geneid'][i][1:])
        
    if len(list(set(map(len,list(df['PFS'])))))==1:
        PFS_len=int(list(set(map(len,list(df['PFS']))))[0])
    else:
        print("error: PFS len")
    if len(list(set(map(len,list(df['seq'])))))==1:   
        sequence_len=int(list(set(map(len,list(df['seq']))))[0])
    else:
        print("error: sequence len")
    #drop features
    y=np.array(df['logFC'],dtype=float)
    guideids=np.array(list(df['guideid']))
    X=df[["expression_level_early","essentiality","length","gene_pos","gene_pos_percentage","duplex_delta_G","Target_delta_G"]]
    random_features=['expression_level_early',"essentiality","length"]
    fix_features=[i for i in X.columns.values if i not in random_features]
    X_random=X[random_features]
    
    X_fix=X[fix_features]
    clusters=list(df['geneid'])
    ### add sequence columns
    PFS_encoded=np.array(PFS_encoded)
    sequence_encoded=np.array(sequence_encoded)
    X_fix=np.c_[X_fix,PFS_encoded,sequence_encoded]
    nts=['A','U','C','G']
    sequence_features=list()
    PFS_features=list()
    for i in range(PFS_len):
        for j in range(len(nts)):
            fix_features.append('PFS_%s_%s'%(i+1,nts[j]))
            PFS_features.append('PFS_%s_%s'%(i+1,nts[j]))
    for i in range(sequence_len):
        for j in range(len(nts)):
            fix_features.append('sequence_%s_%s'%(i+1,nts[j]))
            sequence_features.append('sequence_%s_%s'%(i+1,nts[j]))
    X_fix=pandas.DataFrame(X_fix,columns=fix_features)
    return X_fix,X_random, y ,fix_features,random_features,clusters,guideids


    
if os.path.isdir(output_file_name)==False:
    os.mkdir(output_file_name)

logging_file= output_file_name+"/log.txt"
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logging_file,format='%(asctime)s - %(message)s', level=logging.INFO)
dataset="./gRNAs.csv"

#data fusion
training_df=pandas.read_csv(dataset,sep="\t")
open(output_file_name + '/log.txt','a').write("training: %s"%str(training_df.shape))
X_guide,X_gene,y,guide_features,gene_features,clusters,guideids=DataFrame_input(training_df)

open(output_file_name + '/log.txt','a').write("Training sets: %s\n" % dataset)
open(output_file_name + '/log.txt','a').write("Number of fixed-effect features: %s\n" % len(guide_features))
open(output_file_name + '/log.txt','a').write("Guide features: %s\n" % (guide_features))
open(output_file_name + '/log.txt','a').write("Number of random effect features: %s\n" % len(gene_features))
open(output_file_name + '/log.txt','a').write("random effect features: %s\n" % (gene_features))
open(output_file_name + '/log.txt','a').write("Number of clusters: %s\n" % len(set(clusters)))
estimator=RandomForestRegressor(bootstrap=True, criterion='friedman_mse', max_depth=None, #origin & test
                        max_features=0.22442857329791677, max_leaf_nodes=None,
                        min_impurity_decrease=0.0, 
                        min_samples_leaf=18, min_samples_split=16,
                        min_weight_fraction_leaf=0.0, n_estimators=512, n_jobs=1,
                        verbose=0, warm_start=False,random_state = np.random.seed(111))
guide_features=X_guide.columns.values.tolist()
X_df=pandas.DataFrame(data=np.c_[X_gene,X_guide,y,clusters,guideids],columns=gene_features+guide_features+['log2FC','geneid','guideid'])
X_df = X_df.loc[:,~X_df.columns.duplicated()]
guideid_set=list(set(guideids))
open(output_file_name + '/log.txt','a').write("Estimator:"+str(estimator)+"\n")
dtypes=dict()
for feature in X_df.columns.values:
    if feature != 'geneid':
        dtypes.update({feature:float})
X_df=X_df.astype(dtypes)
evaluations=defaultdict(list)
print(time.asctime(),'Start 10-fold CV...')    
kf=sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=np.random.seed(111))
for train_index, test_index in kf.split(guideid_set):
    guide_train = np.array(guideid_set)[train_index]
    test_index = np.array(guideid_set)[test_index]
    guide_train, guide_val = sklearn.model_selection.train_test_split(guide_train, test_size=0.2,random_state=np.random.seed(111))  
   
    train = X_df[X_df['guideid'].isin(guide_train)]
    y_train=train['log2FC']
    X_train=train[guide_features]
    Z_train=train[gene_features]
    clusters_train=train['geneid']
    
    val = X_df[X_df['guideid'].isin(guide_val)]
    y_val=val['log2FC']
    X_val=val[guide_features]
    Z_val=val[gene_features]
    clusters_val=val['geneid']
    
    ### keep the same test from 3 datasets
    test = X_df[X_df['guideid'].isin(test_index)]
    y_test=test['log2FC']
    X_test=test[guide_features]
    Z_test=test[gene_features]
    clusters_test=test['geneid']
    
    
    mrf_lgbm = MERF(estimator,max_iterations=15)
    mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)
    predictions = mrf_lgbm.predict(X_test, Z_test, clusters_test)        
    spearman_rho,_=spearmanr(np.array(y_test), np.array(predictions))
    evaluations['Rs'].append(spearman_rho)
evaluations=pandas.DataFrame.from_dict(evaluations)
evaluations.to_csv(output_file_name+'/iteration_scores.csv',sep='\t',index=True)

print(time.asctime(),"Training model...")
guide_train, guide_test = sklearn.model_selection.train_test_split(guideid_set, test_size=0.2,random_state=np.random.seed(111))  
guide_train, guide_val = sklearn.model_selection.train_test_split(guide_train, test_size=0.2,random_state=np.random.seed(111))  
train = X_df[X_df['guideid'].isin(guide_train)]
y_train=train['log2FC']
X_train=train[guide_features]
Z_train=train[gene_features]
clusters_train=train['geneid']

val = X_df[X_df['guideid'].isin(guide_val)]
y_val=val['log2FC']
X_val=val[guide_features]
Z_val=val[gene_features]
clusters_val=val['geneid']

test = X_df[X_df['guideid'].isin(guide_test)]
y_test=test['log2FC']
X_test=test[guide_features]
Z_test=test[gene_features]
clusters_test=test['geneid']

mrf_lgbm = MERF(estimator,max_iterations=15)
mrf_lgbm.fit(X_train, Z_train, clusters_train, y_train,X_val, Z_val, clusters_val, y_val)
#SHAP values for fixed-effect model
import shap
print(time.asctime(),"Runing SHAP...")
treexplainer = shap.TreeExplainer(mrf_lgbm.trained_fe_model)
shap_values = treexplainer.shap_values(X_test,check_additivity=False)
shap_values = pandas.DataFrame(shap_values,columns=guide_features)
shap_values.to_csv(output_file_name+"/shap_values_test.csv",index=False,sep='\t')
shap_values.index=X_test.index
values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':guide_features})
values.to_csv(output_file_name+"/shap_value_mean_test.csv",index=False,sep='\t')
sns.set_style("white")
shap.summary_plot(shap_values, X_test, plot_type="bar",show=False,color_bar=True,max_display=10)
plt.subplots_adjust(left=0.35, top=0.95)
plt.savefig(output_file_name+"/shap_value_bar_test.svg",dpi=400)
plt.close()

ax=plt.subplot()
shap.summary_plot(np.array(shap_values), X_test,show=False,max_display=10,alpha=0.1)
plt.subplots_adjust(left=0.4, top=0.95,bottom=0.1)
plt.yticks(fontsize='medium')
plt.xticks(fontsize='medium')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
plt.savefig(output_file_name+"/shap_value_top.svg",dpi=400)
plt.show()
plt.close()    
print(time.asctime(),'Done.')     
open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start)))   