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
import logging
import pandas
import sys
import sklearn.model_selection
import sklearn.metrics
import autosklearn.regression
import autosklearn.metrics
from scipy.stats import spearmanr
import textwrap
start_time=time.time()

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(usage='python %(prog)s datasets [options]',formatter_class=argparse.RawDescriptionHelpFormatter,description=textwrap.dedent('''\
                  This is used to optimize regression models using autosklearn. 
                  
                  Example: python machine_learning_model_optimization_autosklearn.py gRNAs.csv
                  '''))
parser.add_argument("dataset", help="data csv file(s),multiple files separated by ','")
parser.add_argument("-o", "--output", default="results", help="output folder name. default: results")
parser.add_argument("-f","--folds", type=int, default=10, help="Fold of cross validation, default: 5")
parser.add_argument("-e","--ensemble_size", type=int, default=1, help="Ensemble size, default: 50")
parser.add_argument("-t","--test_size", type=float, default=0.2, help="Test size for spliting datasets, default: 0.3")
parser.add_argument("-prt","--per_run_time_limit", type=int, default=360, help="per_run_time_limit (in second), default: 360")
parser.add_argument("-ptt","--time_left_for_this_task", type=int, default=3600, help="time_left_for_this_task (in second), default: 3600")
parser.add_argument("-inest","--include_estimators", type=str, default=None, help="estimators to be included in auto-sklearn. Multiple input separated by ','. If None, then include all. Default: None")
parser.add_argument("-inprepro","--include_preprocessors", type=str, default=None, help="preprocessors to be included in auto-sklearn. Multiple input separated by ','. If None, then include all. Default: None")

args = parser.parse_args()
dataset=args.dataset
output_file_name = args.output
folds=args.folds
ensemble_size=args.ensemble_size
per_run_time_limit=args.per_run_time_limit
time_left_for_this_task=args.time_left_for_this_task
include_estimators=args.include_estimators
include_preprocessors=args.include_preprocessors
test_size=args.test_size
### esitmator and preprocessor setting for auto sklearn
if include_estimators != None:
    include_estimators=include_estimators.split(',')
if include_preprocessors != None:
    include_preprocessors=include_preprocessors.split(',')
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
def self_encode(sequence):#one-hot encoding for single nucleotide features
    integer_encoded=np.zeros([len(sequence),4],dtype=np.float64)
    nts=['A','U','C','G']
    for i in range(len(sequence)):
        integer_encoded[i,nts.index(sequence[i])]=1
    sequence_one_hot_encoded = integer_encoded.flatten()
    return sequence_one_hot_encoded

                
def DataFrame_input(df):
    logging_file= open(output_file_name + '/log.txt','a')
    df=df.replace([np.inf, -np.inf], np.nan).dropna(subset=['logFC','expression_level_early']) #remove crRNAs without expression level or logFC data
    df=df[~(df['PFS']=='A')] 
    logging_file.write("Number of guides: %s \n" % df.shape[0])
    ### adding one-hot encoded sequence features
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
    X=df[["expression_level_early","essentiality","geneid","length","gene_pos","gene_pos_percentage","duplex_delta_G","Target_delta_G"]] #select other features for training
    ### feat_type for auto sklearn
    feat_type=[]
    categorical_indicator=['sequence','PAM','PAM_encoded','sequence_encoded','intergenic','coding_strand','essentiality','geneid']
    headers=list(X.columns.values)
    feat_type=['Categorical' if headers[i] in categorical_indicator else 'Numerical' for i in range(len(headers)) ] 
    
    ### add sequence columns
    PFS_encoded=np.array(PFS_encoded)
    sequence_encoded=np.array(sequence_encoded)
    X=np.c_[X,PFS_encoded,sequence_encoded]
    nts=['A','U','C','G']
    for i in range(PFS_len):
        for j in range(len(nts)):
            headers.append('PFS_%s_%s'%(i+1,nts[j]))
            feat_type.append('Categorical')
    for i in range(sequence_len):
        for j in range(len(nts)):
            headers.append('sequence_%s_%s'%(i+1,nts[j]))
            feat_type.append('Categorical')

    for i in range(PFS_len*4+sequence_len*4):
        feat_type.append('Categorical')

    X=np.array(X,dtype=float)
    return X, y,feat_type, headers,genes
    
def Evaluation(output_file_name,X,y,predictions,estimator,headers,name):
    #scores
    output=open(output_file_name+"/log.txt","a")
    spearman_rho,spearman_p_value=spearmanr(y, predictions)
    output.write(name+"\n")
    output.write("spearmann correlation rho: "+str(spearman_rho)+"\n")
    output.write("spearmann correlation p value: "+str(spearman_p_value)+"\n")
    output.write("coefficient R^2: "+str(estimator.score(X,y))+"\n")
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
    
    
def main():
    training_df=pandas.read_csv(dataset,sep="\t")
    X,y,feat_type,headers,genes=DataFrame_input(training_df)
    open(output_file_name + '/log.txt','a').write("Features: %s\n"%headers)
    open(output_file_name + '/log.txt','a').write("Data input Time: %s seconds\n\n" %('{:.2f}'.format(time.time()-start_time))) 
    train_index, test_index= sklearn.model_selection.train_test_split(list(set(genes)), test_size=0.3,random_state=0) #split based on genes
    X_df=pandas.DataFrame(np.c_[X,y,genes],columns=headers+['log2FC','genes'])
    X_train = X_df[X_df['genes'].isin(train_index)]
    y_train=np.array(X_train['log2FC'],dtype=float)
    X_train=np.array(X_train[headers],dtype=float)
    X_test = X_df[X_df['genes'].isin(test_index)]
    y_test=np.array(X_test['log2FC'],dtype=float)
    X_test=np.array(X_test[headers],dtype=float)
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2,random_state=np.random.seed(111))
        
    automl = autosklearn.regression.AutoSklearnRegressor(
            ensemble_size=ensemble_size,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            include_estimators=include_estimators,
            include_preprocessors=include_preprocessors,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': folds},
            tmp_folder=output_file_name+'/autosklearn_regression_example_tmp',
            output_folder=output_file_name+'/autosklearn_regression_example_out',
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
            disable_evaluator_output=False,
            ensemble_memory_limit=1024, ml_memory_limit= 3072,
            ensemble_nbest=50, seed = 1,
            exclude_estimators=None,exclude_preprocessors=None,get_smac_object_callback=None,
            initial_configurations_via_metalearning=25,
            logging_config=None, metadata_directory = None,
            n_jobs= None, smac_scenario_args= None,metric=autosklearn.metrics.r2) #max_models_on_disc not included in version 0.5.2
    

    automl.fit(X_train.copy(), y_train.copy(),feat_type=feat_type)
        
    ## when cross validation is selected, refit is requirred
    automl.refit(X_train.copy(), y_train.copy())
    
    output= open(output_file_name + '/log.txt','a')
    output.write("Get parameters:\n"+str(automl.get_params())+"\n\n")
    output.write("Show models: \n"+str(automl.show_models())+"\n\n")
    output.write("sprint statistics: %s \n\n"% (automl.sprint_statistics()))
        
    predictions=automl.predict(X_test)
    Evaluation(output_file_name,X_test,y_test,predictions,automl,headers,"X_test")

    logging.info("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))    

if __name__ == '__main__':
    open(output_file_name + '/log.txt','a').write("Python script: %s\n"%sys.argv[0])
    open(output_file_name + '/log.txt','a').write("Parsed arguments: %s\n\n"%args)
    main()
    open(output_file_name + '/log.txt','a').write("Execution Time: %s seconds" %('{:.2f}'.format(time.time()-start_time)))    

#%%
