from wine_quality_model.forecast_model import WineQualityPredictor
import pandas as pd
from sklearn.metrics import f1_score


if __name__ == '__main__':
    training_file_name = 'TrainingDataset.csv'
    validation_file_name = 'ValidationDataset.csv'
    test_file_name = None
    predictor = WineQualityPredictor(
        training_file_name=training_file_name,
        validation_file_name=validation_file_name
    )
    predictor.calibrate_model()

    if test_file_name is None:
        X_test = predictor.X_test
        y_test = predictor.y_test
    else:
        test_dataset = WineQualityPredictor.read_dataset(test_file_name)
        X_test = test_dataset.copy(deep=True)
        y_test = X_test.pop('quality')
        X_test = X_test[predictor.X_train.columns]

    predictions = predictor.predict(X_test)
    predictions = pd.DataFrame(predictions, index=y_test.index)[0]
    f1score = f1_score(y_test, predictions, average='weighted')
    print('F1 Score of test data is ', f1score)
