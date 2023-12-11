from wine_quality_model.forecast_model import WineQualityPredictor
import pandas as pd
from sklearn.metrics import f1_score
import sys
import getopt


def main(test_file=None, use_cluster=False):
    training_file_name = 'TrainingDataset.csv'
    validation_file_name = 'ValidationDataset.csv'
    print('Setting up prediction model...')
    predictor = WineQualityPredictor(
        training_file_name=training_file_name,
        validation_file_name=validation_file_name
    )
    print('Calibrating predictor...')
    if use_cluster:
        print('Cluster mode is turned ON')
        predictor.calibrate_model_cluster()
    else:
        print('Cluster mode is turned OFF')
        predictor.calibrate_model()
    print('Calibration complete')
    if test_file is None:
        print('No test file provided. Using pre-built test file')
        X_test = predictor.X_test
        y_test = predictor.y_test
    else:
        print('Reading test dataset')
        test_dataset = WineQualityPredictor.read_dataset(test_file_name)
        X_test = test_dataset.copy(deep=True)
        y_test = X_test.pop('quality')
        print('Running predictions')
        X_test = X_test[predictor.X_train.columns]

    predictions = predictor.predict(X_test)
    predictions = pd.DataFrame(predictions, index=y_test.index)[0]
    f1score = f1_score(y_test, predictions, average='weighted')
    print('F1 Score of test data is ', f1score)


if __name__ == '__main__':
    use_cluster = False
    test_file_name = None

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "t:c", ["test=", "cluster"])
    for opt, arg in opts:
        if opt in ['-t', '--test']:
            test_file_name = arg
        if opt in ['-c', '--cluster']:
            use_cluster = True
    main(test_file_name, use_cluster=use_cluster)
