#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:34:42 2019

@author: yanying
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import os
import time 
import seaborn as sns
import pandas
import sys
from scipy.stats import spearmanr
import sklearn
import sklearn.metrics
import sklearn.model_selection
from sklearn.feature_selection import VarianceThreshold,GenericUnivariateSelect,f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from collections import defaultdict
import shap
import textwrap
import autosklearn
from scipy import sparse
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from autosklearn.pipeline.implementations.SparseOneHotEncoder import SparseOneHotEncoder
from sklearn.preprocessing import OneHotEncoder
import autosklearn.pipeline.implementations.CategoryShift
from autosklearn.pipeline.implementations.MinorityCoalescer import MinorityCoalescer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')
start_time=time.time()

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawDescriptionHelpFormatter,description=textwrap.dedent('''\
                  This is used to evaluate and interpret the model optimized with autosklearn. 
                  
                  Example: python machine_learning_model_interpretation_treeSHAP.py gRNAs.csv
                  '''))
parser.add_argument("dataset", help="library CSV file")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-t","--test_size", type=float, default=0.3, help="Test size for spliting datasets, default: 0.3")
parser.add_argument("--sequence_first", type=bool, default=False, help="For forward-selection, whether to train the model with sequence features first (True/False). default: False")
args = parser.parse_args()
dataset=args.dataset
output_file_name = args.output
test_size=args.test_size
sequence_first=args.sequence_first
try:
    os.mkdir(output_file_name)
except:
    overwrite=input("File exists, do you want to overwrite? (y/n)")
    if overwrite == "y":
        os.system("rm -r %s"%output_file_name)
        os.mkdir(output_file_name)
    elif overwrite =="n":
        output_file_name=input("Please give a new output file name:")
        os.mkdir(output_file_name)
    else:
        print("Please input valid choice..\nAbort.")
        sys.exit()
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
    ### one-hot encoding sequence features
    PFS_encoded=[]
    sequence_encoded=[]

    for i in df.index:
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
    y=np.array(df['logFC'],dtype=float)
    genes=list(df['geneid'])
    X=df[["expression_level_early","essentiality","geneid","length","gene_pos","gene_pos_percentage","duplex_delta_G","Target_delta_G"]]
    ### feat_type for auto sklearn
    headers=list(X.columns.values)
    feat_type=[]
    categorical_indicator=['essentiality','geneid']
    headers=list(X.columns.values)
    feat_type=['Categorical' if headers[i] in categorical_indicator else 'Numerical' for i in range(len(headers)) ] 
    ### add sequence columns
    PFS_encoded=np.array(PFS_encoded)
    sequence_encoded=np.array(sequence_encoded)
    X=np.c_[X,PFS_encoded,sequence_encoded]
    nts=['A','U','C','G']
    sequence_features=list()
    PFS_features=list()
    for i in range(PFS_len):
        for j in range(len(nts)):
            headers.append('PFS_%s_%s'%(i+1,nts[j]))
            PFS_features.append('PFS_%s_%s'%(i+1,nts[j]))
    for i in range(sequence_len):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
            sequence_features.append('sequence_%s_%s'%(i+1,nts[j]))
    for i in range(PFS_len*4+sequence_len*4):
        feat_type.append('Categorical')

    X=pandas.DataFrame(X,columns=headers)
    return X, y ,headers,feat_type,sequence_features,PFS_features,genes

def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, SparseOneHotEncoder) or isinstance(estimator, OneHotEncoder) :
            # handling all vectorizers
            return [feature_in[int(f.split("_")[0][1:])]+"_"+f.split("_")[1] for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, VarianceThreshold):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in

def get_ct_feature_names(ct,features):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []
    for name, estimator, features in ct.transformers_:
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
    return output_features    



def Evaluation(output_file_name,X,y,predictions,estimator,headers,name):
    #scores
    output=open(output_file_name+"/result.txt","a")
    spearman_rho,spearman_p_value=spearmanr(y, predictions)
    output.write(name+"\n")
    output.write("spearmann correlation rho: "+str(spearman_rho)+"\n")
    output.write("spearmann correlation p value: "+str(spearman_p_value)+"\n")
    output.write("coefficient R^2: "+str(estimator.score(X,y))+"\n")
    output.write("r2: "+str(sklearn.metrics.r2_score(y,predictions))+"\n")
    output.write("explained_variance_score score of "+name+" :"+str(sklearn.metrics.explained_variance_score(y, predictions))+"\n")
    output.write("Mean absolute error regression loss score of "+name+" :"+str(sklearn.metrics.mean_absolute_error(y, predictions))+"\n")
    y=np.array(y)
    
    plt.figure() 
    sns.set_palette("PuBu",2)
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    ax_main.scatter(y,predictions,edgecolors='white',alpha=0.8)
    ax_main.set(xlabel='Experimental log2FC',ylabel='Predicted log2FC')
    ax_xDist.hist(y,bins=70,align='mid',alpha=0.7)
    ax_xDist.set(ylabel='count')
    ax_xDist.tick_params(labelsize=6,pad=2)
    ax_yDist.hist(predictions,bins=70,orientation='horizontal',align='mid',alpha=0.7)
    ax_yDist.set(xlabel='count')
    ax_yDist.tick_params(labelsize=6,pad=2)
    ax_main.text(0.55,0.03,"Spearman R: {0}".format(round(spearman_rho,2)),transform=ax_main.transAxes,fontsize=10)
    plt.savefig(output_file_name+'/'+name+'_scatterplot.png',dpi=300)
    plt.close()
    plt.figure() 
    
def gini_importance(estimator,new_headers):
    fi=pandas.DataFrame({"features":new_headers,"importance":estimator.feature_importances_})
    fi=fi.sort_values(by=["importance"],ascending=False)
    fi.to_csv(output_file_name+ '/gini_importance.csv',sep='\t',index=False)
    plt.figure()
    plt.subplots_adjust(bottom=0.45, top=0.96,right=0.96)
    plt.bar(fi['features'][:30],fi['importance'][:30])
    plt.ylabel('Feature weights')
    plt.xlabel('Top %s Features (total %s)' %(min(30,len(new_headers)),len(new_headers)), fontsize=8)
    plt.xticks(rotation=90,fontsize='x-small')
    plt.savefig( output_file_name+ '/gini_importance.png',dpi=400)    
    plt.close()
    
def SHAP(estimator,X_train,headers):
    X_train=pandas.DataFrame(X_train,columns=headers)
    X_train=X_train.astype(float)
    explainer=shap.TreeExplainer(estimator)
    shap_values =explainer.shap_values(X_train)
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep="\t")
    values=values.sort_values(by='shap_values',ascending=False)
    shap.summary_plot(shap_values, X_train, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.svg")
    plt.close()
    plt.figure()
    shap.summary_plot(shap_values, X_train,show=False,max_display=10,alpha=0.05)
    plt.subplots_adjust(left=0.45, top=0.95)
    plt.yticks(fontsize='medium',fontweight='bold')
    plt.savefig(output_file_name+"/shap_value_top10.svg")
    plt.close()
    values=pandas.DataFrame(data=shap_values,columns=headers)
    values.to_csv(output_file_name+"/shap_value.csv",index=True,sep="\t")
    
def main():
    training_df=pandas.read_csv(dataset,sep="\t")
    X,y,headers,feat_type,sequence_features,PFS_features,genes=DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Features: %s\n"%headers)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time)))  
    
    category_coalescence=MinorityCoalescer(minimum_fraction=0.003623240287266379)
    numerical_imputer=SimpleImputer(add_indicator=False, copy= False, fill_value= None, missing_values= np.nan, strategy='median', verbose= 0)
    numerical_scaler= RobustScaler(with_centering=True, with_scaling=True, quantile_range=(0.2230614629084614, 0.8134431529581335), copy=True)
    
    numerical_transformer=Pipeline(steps=[["imputation", numerical_imputer],
        ["variance_threshold", VarianceThreshold(threshold=0.0)],
        ["rescaling", numerical_scaler]])
    preprocessor=GenericUnivariateSelect(score_func=f_regression, param=0.09849735185327581, mode='fdr')
    
    ###forward-selection for feature importance (modified on 23.04.2021)
    print(time.asctime(),'Start forward-selection...')     
    to_test_features=["expression_level_early","essentiality","geneid","length","gene_pos","gene_pos_percentage","duplex_delta_G","Target_delta_G"]
    to_test_sets=list()
    if sequence_first:
        to_test_sets.append(sequence_features)
        to_test_sets.append(sequence_features+PFS_features)
        for i  in range(1,len(to_test_features)+1):
            to_test_sets.append([sequence_features+PFS_features+to_test_features[:i]])
    else:
        for i  in range(1,len(to_test_features)+1):
            to_test_sets.append(to_test_features[:i])
        to_test_sets.append(to_test_features+sequence_features)
        to_test_sets.append(to_test_features+sequence_features+PFS_features)
    evaluations=defaultdict(list)
    for headers in to_test_sets:
        print(time.asctime(),'Start training with feature set: %s...'%headers)    
        X_sub=X[headers]
        X_sub=X_sub.astype(float)
        open(output_file_name + '/log.txt','a').write('%s headers: %s\n'%((to_test_sets.index(headers)+1),headers))
        categorical_transformer=Pipeline(steps=[["category_shift", autosklearn.pipeline.implementations.CategoryShift.CategoryShift()],
            ["imputation", SimpleImputer(strategy='constant', fill_value=2, copy=False)],
            ["category_coalescence", category_coalescence]
            ]) #
        feat_type=[]
        numerical_indicator=["expression_level_early","length","gene_pos","gene_pos_percentage","duplex_delta_G","Target_delta_G"]
        feat_type=['Categorical' if headers[i] not in numerical_indicator else 'Numerical' for i in range(len(headers)) ] 
        if 'Categorical' not in feat_type:
            sklearn_transf_spec = [
            ["numerical_transformer", numerical_transformer, [headers[i] for i in range(len(feat_type)) if feat_type[i]=='Numerical']]
            ]
        elif 'Numerical' not in feat_type:
            sklearn_transf_spec = [
            ["categorical_transformer", categorical_transformer, [headers[i] for i in range(len(feat_type)) if feat_type[i]=='Categorical']]
            ]
        else:
            sklearn_transf_spec = [
            ["categorical_transformer", categorical_transformer, [headers[i] for i in range(len(feat_type)) if feat_type[i]=='Categorical']],
            ["numerical_transformer", numerical_transformer, [headers[i] for i in range(len(feat_type)) if feat_type[i]=='Numerical']]
            ]
        column_transformer = ColumnTransformer(transformers=sklearn_transf_spec,sparse_threshold=float(sparse.issparse(X_sub)))
        X_sub=column_transformer.fit_transform(X_sub)
        X_preprocessed=preprocessor.fit_transform(X_sub,y)
        
        processed_headers=get_ct_feature_names(column_transformer,headers)
        mask=preprocessor.get_support()
        if False not in mask:
            preprocessed_headers=processed_headers
        else:
            preprocessed_headers=[]
            for i in range(len(mask)):
                if mask[i]==True:
                    preprocessed_headers.append(processed_headers[i])
        processed_headers=preprocessed_headers
        X_df=pandas.DataFrame(np.c_[X_preprocessed,y,genes],columns=processed_headers+['log2FC','genes'])
        print(time.asctime(),'Start 10-fold CV...')     
        ### 10-fold Cross-validation
        kf=sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=np.random.seed(111))
        for train_index, test_index in kf.split(list(set(genes))):
            X_train = X_df[X_df['genes'].isin(train_index)]
            y_train=X_train['log2FC']
            X_train=X_train[processed_headers]
            X_test = X_df[X_df['genes'].isin(test_index)]
            y_test=X_test['log2FC']
            X_test=X_test[processed_headers]
            estimator=HistGradientBoostingRegressor(loss='least_squares',learning_rate=0.033744850983537196,
                    max_iter=512,
                    min_samples_leaf=116,
                    max_depth=None,
                    max_leaf_nodes=27,
                    max_bins=255,
                    l2_regularization=5.5749535239648375e-06,
                    tol=1e-07,scoring='loss',
                    n_iter_no_change=0,
                    validation_fraction=None,verbose=0,warm_start=False,random_state=np.random.seed(111))
            estimator.fit(X_train, y_train)
            predictions=estimator.predict(X_test)
            evaluations['feature_set'].append(to_test_sets.index(headers)+1)
            evaluations['Spearman correlation'].append(spearmanr(y_test,predictions)[0])
    evaluations=pandas.DataFrame.from_dict(evaluations) 
    evaluations.to_csv(output_file_name+"/evaluations.csv",sep='\t',index=False)
    if sequence_first:     
        labels=["gRNA sequence","+PFS sequence","+gene expression level","+essentiality","+geneid","+gene length","+targeting position","+percent targeting position","+gRNA structure ","+mRNA structure"]    
    else:
        labels=["gene expression level","+essentiality","+geneid","+gene length","+targeting position","+percent targeting position","+gRNA structure ","+mRNA structure","+gRNA sequence","+PFS sequence"]  
    
    sns.boxplot(data=evaluations,x='feature_set',y='Spearman correlation',order=range(len(to_test_sets)),color='skyblue')
    plt.xticks(range(len(labels)),labels,rotation=90,fontsize='x-small')
    plt.xlabel("")
    plt.subplots_adjust(bottom=0.3)
    plt.ylim(0,0.8)
    plt.savefig(output_file_name+"/"+'Spearman correlation'+".png",dpi=400)
    plt.close()
    print(time.asctime(),'Start model interpretation using SHAP...')         
    #interprete the model with all features
    SHAP(estimator,X_test,processed_headers) 
    print(time.asctime(),'Done.')     
if __name__ == '__main__':
    open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
    open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))    

#%%
