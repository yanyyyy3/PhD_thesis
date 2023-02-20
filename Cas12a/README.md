# Analysis of CRISPR-Cas12a genome-wide screen in Escherichia coli

## For screen analysis:
To obtain the differential abundance (depletion score) of crRNAs, run *"screen_analysis_differential_abundance.R"* . The result file *"*_QLFTest.csv"* is included here.


## For applying machine learning:
1. To optimize machine learning model using automated machine learning tool auto-sklearn, run *"autosklearn_regressor.py"*.
2. To evaluate and interprete the optimized model from auto-sklearn, run *"sklearn_regressor.py"*.

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


