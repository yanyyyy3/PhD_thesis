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
import itertools
import os
import time 
import seaborn as sns
import logging
import pandas
import sys
import sklearn.model_selection
import sklearn.metrics
import autosklearn.regression
import autosklearn.classification
import autosklearn.metrics
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from autosklearn.pipeline.implementations.SparseOneHotEncoder import SparseOneHotEncoder
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
import pickle
import shap
from scipy.stats import spearmanr,pearsonr
from collections import defaultdict
start_time=time.time()
import warnings
warnings.filterwarnings('ignore')
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawTextHelpFormatter,description="""
This is used to evaluate models optimized using auto-sklearn (tested version 0.14.6).

ensemble_size, folds, per_run_time_limit, time_left_for_this_task, include_estimators, and include_preprocessors are parameters for auto-sklearn. More description please check the API of auto-sklearn (https://automl.github.io/auto-sklearn/master/api.html)

Example: python sklearn_regressor.py gRNAs.csv
                  """)
parser.add_argument("data_csv", help="data csv file")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-H","--headers", type=str, default=None, help="Saved headers from autosklearn, default: None")
parser.add_argument("-d","--data_preprocessor", type=str, default=None, help="Saved data preprocessor from autosklearn, default: None")
parser.add_argument("-r","--regressor", type=str, default=None, help="Saved regressor from autosklearn, default: None")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 10")
parser.add_argument("-t","--timepoint", type=str, default="st4_st1", help="the timepoint of logFC")
args = parser.parse_args()
data_csv=args.data_csv
output_file_name = args.output
folds=args.folds
saved_headers=args.headers
saved_data_preprocessor=args.data_preprocessor
regressor=args.regressor
timepoint=args.timepoint
nts=['A','T','C','G']

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


def self_encode(sequence):
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','T','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

def dinucleotide(sequence):
    nts=['A','T','C','G']
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    encoded=np.zeros([(len(nts)**2)*(len(sequence)-1)],dtype=np.float64)
    for nt in range(len(sequence)-1):
        encoded[nt*len(nts)**2+dinucleotides.index(sequence[nt]+sequence[nt+1])]=1
    return encoded


def DataFrame_input(df):
    ###keep essential genes
    df=df[(df['gene_essentiality']==1)]#&(df['coding_strand']==1)&(df['intergenic']==0)
    df=df.dropna(subset=['%s_logFC'%timepoint])
    
    open(output_file_name + '/log.txt','a').write("Number of selected guides: %s\n" % df.shape[0])
    open(output_file_name + '/log.txt','a').write("Number of targeting genes: %s\n" % len(list(set(df['geneid']))))
    y=np.array(df['%s_logFC'%timepoint],dtype=float)
    plt.figure()
    sns.distplot(y,bins=100,kde_kws={"shade":False, "bw":0.1})
    plt.xlabel("%s logFC"%timepoint)
    plt.title("Distribution of logFC")
    plt.savefig(output_file_name+"/%s_logFC_dist.png"%timepoint)
    plt.close()
    ### adding
    sequences=list(dict.fromkeys(df['sequence']))
    PAM_encoded=[]
    sequence_encoded=[]
    dinucleotide_encoded=[]
    nts=['A','T','C','G']
    for i in df.index:
        PAM_encoded.append(self_encode(df['PAM'][i]))
        sequence_encoded.append(self_encode(df['sequence'][i]))
        dinucleotide_encoded.append(dinucleotide(df['sequence_30nt'][i]))
        df.at[i,'guideid']=sequences.index(df['sequence'][i])
        df.at[i,'geneid']=int(df['geneid'][i][1:])
    guideids=np.array(list(df['guideid']))
    geneids=np.array(list(df['geneid']))
    if len(list(set(map(len,list(df['PAM'])))))==1:
        PAM_len=int(list(set(map(len,list(df['PAM']))))[0])
    else:
        print("error: PAM len")
    if len(list(set(map(len,list(df['sequence'])))))==1:   
        sequence_len=int(list(set(map(len,list(df['sequence']))))[0])
    else:
        print("error: sequence len")
    if len(list(set(map(len,list(df['sequence_30nt'])))))==1:   
        dinucleotide_len=int(list(set(map(len,list(df['sequence_30nt']))))[0])
    else:
        print("error: sequence len")
    ## remove description columns    

    feat_type=[]
    ### add sequence columns
    PAM_encoded=np.array(PAM_encoded)
    sequence_encoded=np.array(sequence_encoded)
    dinucleotide_encoded=np.array(dinucleotide_encoded)
    X=np.c_[sequence_encoded,PAM_encoded,dinucleotide_encoded]
    headers=[]
    nts=['A','T','C','G']
    for i in range(sequence_len):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
    for i in range(PAM_len):
        for j in range(len(nts)):
            headers.append('PAM_%s_%s'%(i+1,nts[j]))
    items=list(itertools.product(nts,repeat=2))
    dinucleotides=list(map(lambda x: x[0]+x[1],items))
    for i in range(dinucleotide_len-1):
        for dint in dinucleotides:
            headers.append(dint+str(i+1)+str(i+2))
          
    for i in range(PAM_len*4+sequence_len*4+(dinucleotide_len-1)*4*4):
        feat_type.append('Categorical')
        
    X=pandas.DataFrame(X,columns=headers)      
    open(output_file_name + '/log.txt','a').write("Number of features: %s\n" % len(headers))
    open(output_file_name + '/log.txt','a').write("Features: "+",".join(headers)+"\n\n")
    return X, y,feat_type, headers,guideids,geneids

def Evaluation(output_file_name,y,predictions,name):
    #scores
    output=open(output_file_name+"/result.txt","a")
    spearman_rho,spearman_p_value=spearmanr(y, predictions)
    pearson_rho,pearson_p_value=pearsonr(y, predictions)
    output.write(name+"\n")
    output.write("spearman correlation rho: "+str(spearman_rho)+"\n")
    output.write("spearman correlation p value: "+str(spearman_p_value)+"\n")
    output.write("pearson correlation rho: "+str(pearson_rho)+"\n")
    output.write("pearson correlation p value: "+str(pearson_p_value)+"\n")
    output.write("r2: "+str(sklearn.metrics.r2_score(y,predictions))+"\n")
    output.write("explained_variance_score score of "+name+" :"+str(sklearn.metrics.explained_variance_score(y, predictions))+"\n")
    output.write("Mean absolute error regression loss score of "+name+" :"+str(sklearn.metrics.mean_absolute_error(y, predictions))+"\n")
    y=np.array(y)
    
    # scatter plot
    plt.figure() 
    sns.set_palette("PuBu",2)
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
    ax_main.scatter(y,predictions,edgecolors='white',alpha=0.8)
    ax_main.set(xlabel='Experimental logFC',ylabel='Predicted logFC')
    ax_xDist.hist(y,bins=70,align='mid',alpha=0.7)
    ax_xDist.set(ylabel='count')
    ax_xDist.tick_params(labelsize=6,pad=2)
    ax_yDist.hist(predictions,bins=70,orientation='horizontal',align='mid',alpha=0.7)
    ax_yDist.set(xlabel='count')
    ax_yDist.tick_params(labelsize=6,pad=2)
    ax_main.text(0.55,0.03,"Spearman R: {0}".format(round(spearman_rho,2)),transform=ax_main.transAxes,fontsize=10)
    ax_main.text(0.55,0.10,"Pearson R: {0}".format(round(pearson_rho,2)),transform=ax_main.transAxes,fontsize=10)
    plt.savefig(output_file_name+'/'+name+'_scatterplot.png',dpi=300)
    plt.close()
    
def SHAP(estimator,X,headers):
    X=pandas.DataFrame(X,columns=headers)
    # X.to_csv(output_file_name+"/shap_samples_scaled.csv",sep='\t',index=False)
    X=X.astype(float)
    explainer=shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X,check_additivity=False)
    values=pandas.DataFrame({'shap_values':np.mean(np.absolute(shap_values),axis=0),'features':headers})
    values.to_csv(output_file_name+"/shap_value_mean.csv",index=False,sep='\t')
    
    shap.summary_plot(shap_values, X, plot_type="bar",show=False,color_bar=True,max_display=10)
    plt.subplots_adjust(left=0.35, top=0.95)
    plt.savefig(output_file_name+"/shap_value_bar.svg",dpi=400)
    plt.close()
    
    for i in [10,15,30]:
        shap.summary_plot(shap_values, X,show=False,max_display=i,alpha=0.5)
        plt.subplots_adjust(left=0.45, top=0.95,bottom=0.2)
        plt.yticks(fontsize='medium')
        plt.xticks(fontsize='medium')
        plt.savefig(output_file_name+"/shap_value_top%s.svg"%(i),dpi=400)
        plt.savefig(output_file_name+"/shap_value_top%s.png"%(i),dpi=400)
        plt.close()    

    # shap_values=pandas.DataFrame(shap_values,columns=headers)
    # shap_values.to_csv(output_file_name+"/shap_values.csv",sep='\t',index=False)
    
def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'preprocessor'):
        estimator=estimator.preprocessor
    else:
        estimator=estimator.choice.preprocessor
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, SparseOneHotEncoder) or isinstance(estimator, OneHotEncoder) :
            # handling all vectorizers
            return [feature_in[int(f.split("_")[0][1:])]+"_"+f.split("_")[1] for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names()
    elif isinstance(estimator, VarianceThreshold):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct,features):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []
    # print(ct.transformers_)
    for name, estimator, features_ind in ct.transformers_:
        # print(name, estimator, features_ind)
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = [features[i] for i in features_ind]
                # print(current_features)
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
                
    return output_features
def main():
    open(output_file_name + '/log.txt','a').write(time.asctime())
    open(output_file_name + '/log.txt','a').write("\nPython script: %s\n"%sys.argv[0])
    open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
    training_df=pandas.read_csv(data_csv,sep="\t")
    training_df = training_df.sample(frac=1,random_state=np.random.seed(111)).reset_index(drop=True)
    logging.info("training input shape: %s"% str(training_df.shape))
    
    X,y,feat_type,headers,guideids,geneids=DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))  
    X_df=pandas.DataFrame(data=np.c_[X,y,guideids,geneids],columns=headers+['logFC','guideid','geneid'])
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    if regressor ==None:
        estimator=HistGradientBoostingRegressor(loss='least_squares',learning_rate=0.0913971028976721,
                            max_iter=512,
                            min_samples_leaf=2,
                            max_depth=None,
                            max_leaf_nodes=9,
                            max_bins=255,
                            l2_regularization=0.005746611563553693,
                            tol=1e-07,scoring='loss',
                            n_iter_no_change=20,
                            validation_fraction=None,verbose=0,warm_start=False,random_state=np.random.seed(111))
        open(output_file_name + '/log.txt','a').write("Estimator:"+str(estimator)+"\n")
    else:
        estimator=pickle.load(open(regressor,'rb'))
        print(estimator.get_params())
        params=estimator.get_params()
        params.update({"random_state":np.random.seed(111),'max_iter':512})
        if 'early_stop' in params.keys():
            params.pop('early_stop', None)
        if params['max_depth']=='None':
            params['max_depth']=None
        estimator=HistGradientBoostingRegressor(**params)
        open(output_file_name + '/log.txt','a').write("regressor:"+str(estimator)+"\n")
    scaler=StandardScaler()
    selector = VarianceThreshold()
    open(output_file_name + '/log.txt','a').write("selector:"+str(selector)+"\n")
    open(output_file_name + '/log.txt','a').write("scaler:"+str(scaler)+"\n")
    
    if saved_data_preprocessor !=None:
        data_preprocessor=pickle.load(open(saved_data_preprocessor,'rb'))
        print(data_preprocessor.choice.config)
        open(output_file_name + '/log.txt','a').write("data_preprocessor:"+str(data_preprocessor.choice.config)+"\n")
    # guideid_set=list(set(guideids))
    ##split the combined training set into train and test
    #k-fold cross validation
    evaluations=defaultdict(list)
    kf=sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=np.random.seed(111))
    guideid_set=list(set(guideids))
    fold=0
    for train_index, test_index in kf.split(guideid_set):##split the combined training set into train and test based on guideid
        train_index=np.array(guideid_set)[train_index]
        test_index=np.array(guideid_set)[test_index]
        train = X_df[X_df['guideid'].isin(train_index)]
        y_train=train['logFC']
        X_train=train[headers]
        X_train=selector.fit_transform(X_train)
        X_train=scaler.fit_transform(X_train)
        
        test = X_df[X_df['guideid'].isin(test_index)]
        y_test=test['logFC']
        X_test=test[headers]
        
        X_test=selector.transform(X_test)
        X_test=scaler.transform(X_test)
    
        if saved_data_preprocessor !=None:
            data_preprocessor.fit(X_train,y_train)
            X_train=data_preprocessor.transform(X_train)
            X_test=data_preprocessor.transform(X_test)
        print("Fold {0}: train {1} samples targeting {2} genes, test {3} samples targeting {4} genes.  ".format(fold+1,X_train.shape[0],len(set(train['geneid'])),X_test.shape[0],len(set(test['geneid']))))
        open(output_file_name + '/log.txt','a').write("Fold {0}: train {1} samples targeting {2} genes, test {3} samples targeting {4} genes.  \n".format(fold+1,X_train.shape[0],len(set(train['geneid'])),X_test.shape[0],len(set(test['geneid']))))
        estimator = estimator.fit(np.array(X_train,dtype=float),np.array(y_train,dtype=float))
        predictions = estimator.predict(np.array(X_test,dtype=float))
        print(spearmanr(y_test, predictions,nan_policy='omit')[0])
        fold+=1
        evaluations['Rs'].append(spearmanr(y_test, predictions)[0])
        
        
    evaluations=pandas.DataFrame.from_dict(evaluations)
    evaluations.to_csv(output_file_name+'/iteration_scores.csv',sep='\t',index=True)
    
    
    eva=evaluations[['Rs']]
    eva.boxplot()
    plt.xticks(fontsize=12)
    plt.ylabel('score')
    plt.savefig(output_file_name+'/cv_Rs.png')
    plt.savefig(output_file_name+'/cv_Rs.svg')
    plt.close()
    ### save model trained with all guides
    X_all=X_df[headers]
    y_all=X_df['logFC']
    # X_all.to_csv(output_file_name+"/shap_samples_nonscaled.csv",sep='\t',index=False)
    X_all=selector.fit_transform(X_all)
    mask=selector.get_support()
    if False not in mask:
        new_headers=headers
    else:
        if len(mask)==len(headers):
            new_headers=[]
            for i in range(len(mask)):
                if mask[i]:
                    new_headers.append(headers[i])
    print(len(new_headers))
    X_all=scaler.fit_transform(X_all)
    if saved_data_preprocessor !=None:
        # feat_type=dict()
        # for i in range(len(new_headers)):
        #     if new_headers[i] not in numeric_indicator:
        #         feat_type.update({i:"Categorical"})
        #     else:
        #         feat_type.update({i:"Numerical"})
        data_preprocessor=pickle.load(open(saved_data_preprocessor,'rb'))
        # data_preprocessor.choice.feat_type=feat_type
        data_preprocessor.fit(X_all,y_all)
        # print(dir(data_preprocessor))
        X_all=data_preprocessor.transform(X_all)
        new_headers=get_ct_feature_names(data_preprocessor.choice.column_transformer,new_headers)
        print(len(new_headers))
    print(X_all.shape)
    estimator = estimator.fit(np.array(X_all,dtype=float),np.array(y_all,dtype=float))
    if os.path.isdir(output_file_name+'/saved_model')==False:  
        os.mkdir(output_file_name+'/saved_model')
    pickle.dump(selector, open(output_file_name+'/saved_model/selector.sav', 'wb'))
    pickle.dump(scaler, open(output_file_name+'/saved_model/scaler.sav', 'wb'))
    pickle.dump(estimator, open(output_file_name+'/saved_model/estimator.sav', 'wb'))
    pickle.dump(headers, open(output_file_name+'/saved_model/headers.sav', 'wb'))
    
    SHAP(estimator,X_all,new_headers)
    Evaluation(output_file_name,y_all,estimator.predict(np.array(X_all,dtype=float)),"X_all")
    
if __name__ == '__main__':
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds\n" %('{:.2f}'.format(time.time()-start_time)))    
    open(output_file_name + '/log.txt','a').write(time.asctime())
#%%
