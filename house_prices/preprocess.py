import pandas as pd
import numpy as np
from joblib import dump


ENCODER_PATH= "../models/encoder.joblib"
SCALAR_PATH="../models/scalar.joblib"  

def drop_unwanted_columns(data: pd.DataFrame, to_remove_columns: list=[]) -> pd.DataFrame:
    return data.drop(to_remove_columns, axis = 1);

def encode_categorical_features(encoder,data: pd.DataFrame,is_test:bool =False) -> pd.DataFrame:
    data_categorical = data.select_dtypes(include=['object']).columns
    if not is_test :
        encoder.fit(data[data_categorical])
        dump(encoder ,ENCODER_PATH)
    data[data_categorical]=encoder.transform(data[data_categorical])
    return data;

def fill_features_nulls(data: pd.DataFrame) -> pd.DataFrame:
    
    data_numerical= data.select_dtypes([np.int64,np.float64]).columns
    data_categorical = data.select_dtypes(include=['object']).columns

    data[data_numerical]=data[data_numerical].fillna(data[data_numerical].mean())
    
    for feature in data_categorical:
        data[feature].interpolate(method ='linear', limit_direction ='forward', inplace=True)
        data[feature].interpolate(method ='linear', limit_direction ='backward',inplace=True)
        
    return data;

def scale_data(scalar,data: pd.DataFrame,is_test:bool =False) -> pd.DataFrame:
    if not is_test:
        scalar.fit(data)
        dump(scalar ,SCALAR_PATH)
    return pd.DataFrame(scalar.transform(data),columns = data.columns)

def data_preprocessing(data: pd.DataFrame,encoder,scalar,is_test:bool=False) -> pd.DataFrame:  
   
    # Carefully Selected Features (after analysis)
    list_of_features =["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","1stFlrSF",
                       "FullBath","YearBuilt","YearRemodAdd","BsmtFinSF1","Foundation",
                       "LotFrontage","WoodDeckSF","MasVnrArea","Fireplaces",
                       "ExterQual","BsmtQual","KitchenQual","GarageFinish",
                       "GarageType","HeatingQC"]

    # To remove Columns 
    unwanted_columns = ["PoolQC", "MiscFeature","Alley","Fence","FireplaceQu"]
    
    data= drop_unwanted_columns(data,to_remove_columns=unwanted_columns)
    
    data =data[list_of_features]
    
    data= encode_categorical_features(encoder,data,is_test)
    
    data= fill_features_nulls(data)
    
    data= scale_data(scalar,data,is_test)
    
    return data
