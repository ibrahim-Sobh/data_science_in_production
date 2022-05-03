import pandas as pd
import numpy as np
from house_prices.preprocess import data_preprocessing
from joblib import load

ENCODER_PATH = "../models/encoder.joblib"
SCALAR_PATH = "../models/scalar.joblib"
MODEL_PATH = "../models/model.joblib"


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:

    # load the encoder
    ordinal = load(ENCODER_PATH)

    # load the scalar
    scalar = load(SCALAR_PATH)

    # load the model
    model = load(MODEL_PATH)

    input_data = data_preprocessing(
        input_data, encoder=ordinal, scalar=scalar, is_test=True)

    # Validation-set evaluation
    y_predictions = model.predict(input_data)

    return y_predictions
