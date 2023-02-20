# Analysis of CRISPR-Cas13a genome-wide screen in Escherichia coli

##For screen analysis:
1. The sequence data of library screening is available with accession number GSE179913. After merging the pair-end reads, run *"screen_analysis_read_count.py"* to obtain the counts of crRNAs. A summary table of counts from all samples is also available in GSE179913.
2. To obtain the differential abundance (depletion score) of crRNAs, run *"screen_analysis_differential_abundance.R"* . The result file *"targeting-nt_QLFTest.csv"* is included here.
3. To investigate the relationship between expression level and depletion scores for each gene, run *"Depletion_ExpressionLevel_comparison.py"*. Expression level data can be assessed in GSE179914.

##For applying machine learning:
1. To optimize machine learning model using automated machine learning tool auto-sklearn, run *"machine_learning_model_optimization_autosklearn.py"*.
2. To evaluate and interprete the optimized model from auto-sklearn, run *"machine_learning_model_interpretation_treeSHAP.py"* (optimized model has been implemented in the script).
3. To further explore the contribution of guide features to depletion, run *"machine_learning_guide_contribution_MERF.py"*.

For each Python script, "-h" shows the detailed description, options and example to run the script. 
 

# Requirements

Python scripts were written in version 3.9. To install all Python dependencies, conda is recommended. 


## Python packages

  |Name             |      Version       |           
  |-----------------|--------------------|
  |python           |       3.9.12       | 
  |auto-sklearn     |       0.14.6       | 
  |scikit-learn     |       0.24.2       |
  |shap             |       0.39         | 
  |numpy            |       1.21.5       | 
  |merf             |       1.0          |
  |matplotlib       |       3.5.1        |  
  |seaborn          |       0.11.2       |
  |pandas           |       1.4.3        |
  |biopython        |       1.78         | 


