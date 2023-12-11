# wine_quality_prediction

This is a model to predict the quality of wine based on certain features. The model is based on CatBoostClassifier and runs under multiclass mode to predict the quality of wine. CatBoostClassifier is a tree based model that uses gradient boosting. The model parameters such as number of estimators, depth of the trees, learning rate and subsample are optimized using cross validation. By default, the model uses pre-optimized values. However, the user can re-optimize the parameters in case training or validation dataset has changed using run_cross_validation() method in the model.

Training and validation datasets are unbalanced so the model implicitly resamples/subsamples classes using a randomization seed to ensure that the datasets used for calibration and validation are balanced. 

Installation
------------

The model requires python 3.9 or later to run. Please create a clean environment to run the model using the following command.

python -m venv envname

Activate the environment like this.

source ./envname/bin/activate

Once the environment is activated, please use the requirements.txt file to install necessary packages as follows.

pip install -r requirements.txt


Execution
---------

To execute the model the following options are available. When run locally the model takes about 6 mins to calibrate.

1. Run using default test data locally.

   python main.py
   
3. Run using a custom test data locally.

   python main.py -t your_test_data_.csv
   or
   python main.py --test your_test_data.csv
   
5. Run using a default or custom test data on a dask cluster.

   python main.py -t your_test_data.csv -c
   or
   python main.py -t your_test_data.csv --cluster

   python main.py -c
   or
   python main.py --cluster

To run in cluster mode, please ensure that the clusters have been initiated using the following commands on a seperate terminal for scheduler and new EC2 instances for workers.

For scheduler as follows.

dask scheduler 

For workers as follows.

dask worker scheduler_ip:8786

Please make sure that the workers are also built using the same installation instructions.


Output
------

The output is weighted F1 score since data could be unbalanced and there are multiple classes. This is calculated using f1_score from sklearn.

   


