import pandas as pd
import numpy as np
import copy
from catboost import CatBoostClassifier
from dask.distributed import Client
import sklearn


class WineQualityPredictor:
    def __init__(self, training_file_name=None, validation_file_name=None):
        self.training_file_name = training_file_name
        self.validation_file_name = validation_file_name

        # Have been optimized based on grid search cross validation
        # To re-optimize please use run_cross_validation()
        self.model_params = {
            'boosting_type': 'Ordered',
            'loss_function': 'MultiClass',
            'bootstrap_type': 'Bernoulli',
            'silent': True,
            'random_state': 121,
            'depth': 6,
            'n_estimators': 500,
            'learning_rate': 0.1,
            'subsample': 0.85,
            'eval_metric': 'TotalF1',
            'auto_class_weights': 'SqrtBalanced'
        }

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.training_dataset_final = None
        self.validation_dataset_final = None

        self.process_data()

        self.cv_results = None
        self.model = None

        self.scheduler_ip_port = '172.31.23.175:8786'

    def process_data(self):
        training_dataset = self.read_dataset(self.training_file_name)
        validation_dataset = self.read_dataset(self.validation_file_name)

        self.training_dataset_final = training_dataset
        self.validation_dataset_final = validation_dataset
        self.prepare_calibration_data()

    def prepare_calibration_data(self):
        X = self.training_dataset_final.copy()
        y = X.pop('quality')
        X_train, y_train = X, y
        X_test = self.validation_dataset_final.copy()
        y_test = X_test.pop('quality')

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test[X_train.columns]
        self.y_test = y_test

    @staticmethod
    def get_param_grid():
        common_params = {
            'boosting_type': 'Ordered',
            'bootstrap_type': 'Bernoulli',
            'silent': True,
            'random_state': 121,
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1',
            'use_class_weights': 'SqrtBalanced'
        }
        depths = [5, 6, 7, 8, 9, 10]
        n_estimators = [100, 500, 1000, 2000]
        subsamples = [0.55, 0.85]
        learning_rates = [0.01, 0.1]
        search_params = []
        for depth in depths:
            for n_estimator in n_estimators:
                for subsample in subsamples:
                    for learning_rate in learning_rates:
                        params = copy.deepcopy(common_params)
                        params['depth'] = depth
                        params['n_estimators'] = n_estimator
                        params['learning_rate'] = learning_rate
                        params['subsample'] = subsample
                        search_params.append(params)
        return search_params

    def run_cross_validation(self):
        search_params = WineQualityPredictor.get_param_grid()
        X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test
        cv_results = []
        scores = []
        for params in search_params:
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
            pred = model.predict(X_test)
            pred = pd.DataFrame(pred)[0]
            score = sklearn.metrics.f1_score(y_test, pred, average='weighted')
            cv_results.append({'params': params, 'score': score})
            scores.append(score)
        self.cv_results = cv_results
        best_params = cv_results[np.argmax(scores)]['params']
        self.model_params = best_params

    def calibrate_model(self):
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        model = CatBoostClassifier(**self.model_params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True, use_best_model=True)
        self.model = model

    def calibrate_model_cluster(self):
        client = Client(self.scheduler_ip_port)
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        model = CatBoostClassifier(**self.model_params)
        with client:
            model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
        self.model = model

    def predict(self, X_test):
        if self.model is None:
            print('Model needs to be calibrated first!')
        else:
            return self.model.predict(X_test)

    @staticmethod
    def read_dataset(file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        header = lines[0]
        column_names = header.split(';')
        columns = []
        for col in column_names:
            name = col.split('\n')[0]
            name = name.strip('\"')
            columns.append(name)
        n_lines = len(lines)
        all_line_values = []
        for idx in range(1, n_lines):
            line = lines[idx]
            line_values = []
            values = line.split(';')
            for val in values:
                line_values.append(float(val.split('\n')[0]))
            all_line_values.append(line_values)
        return pd.DataFrame(all_line_values, columns=columns)
