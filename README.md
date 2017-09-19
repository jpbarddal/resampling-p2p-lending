resampling-p2p-lending
# Improving Credit Risk Prediction in Online Peer-to-Peer (P2P) Lending Using Imbalanced Learning Techniques
This repository contains all scripts used in the experiments of this work. The Python version was 3.6 and the base frameworks used for machine learning tasks and resampling were
* [Scikit](http://scikit-learn.org)
* [Imbalanced-learn](http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html)



Also, two classes were build in order to deal with cross validation in the presence of resampling: imbalance.crossValidationStratified and imbalance.classifyCrossValidation.
This experiment was divided into two parts: preparation of the data and, finally, classification.

## data_preparation


## experiment_scripts
In the experiments, the major techniques to handle class imbalance were used. Each one of this techniques have a specific script. 

### baseline_p2p.py
This script calculates the baseline for the experiments. At first, the test and train data sets are defined, followed by the classifiers. Then, for each classifier the train data set is used to find the best parameters, using 5 fold cross validation. The final step consists of test the tunnel trained classifier at the test set, using 5 fold cross validation. 
To run the script: python3.6 baseline_p2p.py > log_baseline.log 2> error_baseline.log

### ensembles_p2p.py
The ensemble script execute the same steps that the baseline script.
To run the script: python3.6 ensembles_p2p.py > log_baseline.log 2> error_baseline.log

### sampling_p2p_1.py
To run the script: python3.6 sampling_p2p_1.py > log_baseline.log 2> error_baseline.log

### sampling_p2p_5.py
To run the script: python3.6 sampling_p2p_5.py > log_baseline.log 2> error_baseline.log