import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class Preprocessing:
    def __init__(self, plan=None, X_train=None, X_test=None, y_train=None):
        self.plan = plan
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        
        self.columns_to_drop = plan["id_cols"]
        self.log_transform_cols = plan["skewed_cols"]
        self.robust_scale_cols = plan["outlier_cols"]
        self.ohe_encode_cols = plan["low_card_cols"]
        self.high_card_cols = plan["high_card_encode"]

    def get_categoric_cols(self, dataframe):
        return dataframe.select_dtypes(include=['object', 'bool', 'category']).columns.to_list()
    
    def get_numeric_cols(self, dataframe):
        return dataframe.select_dtypes(include=[np.number]).columns.to_list()
        
    def prepare_data(self):
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        
        to_drop = [c for c in self.columns_to_drop if c in X_train.columns]
        
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
        
        log_cols = [c for c in self.log_transform_cols if c in X_train.columns]
        robust_cols = [c for c in self.robust_scale_cols if c in X_train.columns]
        ohe_cols = [c for c in self.ohe_encode_cols if c in X_train.columns]
        standard_cols = [c for c in self.get_numeric_cols(X_train) if c not in log_cols + robust_cols]
        
        log_pipeline = Pipeline([
            ('impute-num', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(lambda x: np.log1p(np.clip(x, a_min=0, a_max=None)))),
            ('scaler', StandardScaler())
        ])
        
        robust_pipeline = Pipeline([
            ('impute-num', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        standard_pipeline = Pipeline([
            ('impute-num', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        ohe_pipeline = Pipeline([
            ('impute-cat', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        self.column_transformer = ColumnTransformer([
            ('log', log_pipeline, log_cols),
            ('robust', robust_pipeline, robust_cols),
            ('standard', standard_pipeline, standard_cols),
            ('ohe', ohe_pipeline, ohe_cols)
        ])
        
        return X_train, X_test