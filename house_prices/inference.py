import pandas as pd
import numpy as np
from house_prices.preprocess import data_preprocessing
from joblib import load

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    
    # load the encoder 
    encoder_filename="../models/encoder.joblib" 
    ordinal =load(encoder_filename)

    # load the scalar
    scalar_filename="../models/scalar.joblib" 
    scalar=load(scalar_filename)

    # load the model
    model_filename="../models/model.joblib" 
    model= load(model_filename)
    
    input_data=data_preprocessing(input_data,encoder=ordinal,scalar=scalar,is_test=True)
    
    #Validation-set evaluation
    y_predictions=model.predict(input_data)
    
    return y_predictions