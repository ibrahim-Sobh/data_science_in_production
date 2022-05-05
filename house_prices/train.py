
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from house_prices.preprocess import data_preprocessing
from joblib import dump

MODEL_PATH = "../models/model.joblib"


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray,
                  precision: int = 3) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return round(rmsle, precision)


def evaluate_performance(y_pred: np.ndarray, y_true: np.ndarray,
                         precision: int = 2, comment: str = "")\
                          -> dict[str, str]:
    y_pred = y_pred.ravel()
    y_pred = abs(y_pred)

    y_true = y_true.ravel()

    rmse = compute_rmsle(y_true, y_pred, precision)
    key = comment+"_rmse"

    return dict({key: rmse})


def data_split_test_train_validation(data: pd.DataFrame, test_size: int = 0.2,
                                     validation_size: int = 0.2)\
                                      -> pd.DataFrame:
    # Split Train / Test
    X = data.loc[:, data.columns != 'SalePrice']
    y = data.SalePrice

    # First Split L between Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)
    # Second Split :between Train and Validation
    X_train, X_validation, y_train, y_validation = \
        train_test_split(X_train, y_train, test_size=validation_size,
                         random_state=42)
    # return all splitted data sets ( 6 sets )
    return X_train, X_test, X_validation, y_train, y_test, y_validation


def build_model(data: pd.DataFrame) -> dict[str, str]:

    # split data into Train, Test, and Validation
    X_train, X_test, X_validation, y_train, y_test, y_validation = \
        data_split_test_train_validation(data)

    # Preprocessing(cleaning data and training encoders,scalars)
    X_train = data_preprocessing(X_train, is_test=False)

    # Preprocessing(cleaning data and using trained encoders,scalars)
    X_validation = data_preprocessing(X_validation, is_test=True)

    # Define an evaluation dictonary
    evaluations_dict = dict()

    # Defining the Machine Learning model
    LR_model = LinearRegression()

    # Train model
    LR_model.fit(X_train, y_train)
    dump(LR_model, MODEL_PATH)

    # Validation-set evaluation
    y_valid_predictions = LR_model.predict(X_validation)
    validation_evaluation = evaluate_performance(y_pred=y_valid_predictions,
                                                 y_true=y_validation,
                                                 precision=3,
                                                 comment="Validation")
    evaluations_dict.update(validation_evaluation)

    # Model Build Evalution on Testing Set
    # -------------------------------------
    # Preprocessing(cleaning data and using trained encoders,scalars)
    X_test = data_preprocessing(X_test, is_test=True)

    # Testing-set evaluation
    y_test_predictions = LR_model.predict(X_test)
    test_evaluation = evaluate_performance(y_pred=y_test_predictions,
                                           y_true=y_test, precision=3,
                                           comment="Test")
    evaluations_dict.update(test_evaluation)
    # Returns a dictionary with the model performances
    # (for example {"rmse": 0.18})
    return evaluations_dict
