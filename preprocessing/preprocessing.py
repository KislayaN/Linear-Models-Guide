import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

class Preprocessing:
    def __init__(self, dataframe=None, X_train=None, X_test=None, y_train=None, y_test=None, feature_cols=None, columns_to_drop=None, vif_selected_cols=None, robust_scale_cols=None, log_transform_cols=None, standard_scale_cols=None, ohe_encode_cols=None):
        self.dataframe = dataframe
        self.feature_names = feature_cols
        self.columns_to_drop = columns_to_drop
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.vif_selected_cols = vif_selected_cols
        self.log_transform_cols = log_transform_cols
        self.robust_scale_cols = robust_scale_cols
        self.standard_scale_cols = standard_scale_cols
        self.ohe_encode_cols = ohe_encode_cols
        
    def get_categoric_cols(self, dataframe):
        return dataframe.select_dtypes(include=['object', 'bool']).columns.to_list()
    
    def get_numeric_cols(self, dataframe):
        return dataframe.select_dtypes(include=['float', 'int']).columns.to_list()
    
    def target_encode(self, X_train=None, y_train=None, high_card_cols=None, X_test=None):
        global_mean = y_train.mean()
    
        for col in high_card_cols:
            mapping = pd.concat([X_train[col], y_train], axis=1).groupby(col)[y_train.name].mean()
            
            if X_train is not None:
                X_train[col] = X_train[col].map(mapping)
                
            if X_test is not None:
                X_test[col] = X_test[col].map(mapping)
            
                # handle unseen categories
                X_test[col].fillna(global_mean, inplace=True)
        
        return X_train, X_test
        
    def preprocess(self):
        
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        
        y_train = self.y_train.copy()
        
        X_train = X_train.drop(columns=self.columns_to_drop)
        X_test = X_test.drop(columns=self.columns_to_drop)
        
        X_train = X_train[self.vif_selected_cols]
        X_test = X_test[self.vif_selected_cols]
        
        high_card_encode = [c for c in high_card_encode if c in X_train.columns]
        
        X_train, X_test = self.target_encode(
            X_test=X_test,
            X_train=X_train,
            y_train=y_train,
            high_card_cols=high_card_encode
        )
        
        log_cols = [c for c in self.log_transform_cols if c in X_train.columns]
        robust_cols = [c for c in self.robust_scale_cols if c in X_train.columns]
        standard_cols = [c for c in self.standard_scale_cols if c in X_train.columns]
        ohe_cols = [c for c in self.ohe_encode_cols if c in X_train.columns]
        
        log_pipeline = Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', StandardScaler())
        ])
        
        robust_pipeline = Pipeline([
            ('scaler', RobustScaler())
        ])
        
        standard_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer([
            ('log', log_pipeline, log_cols),
            ('robust', robust_pipeline, robust_cols),
            ('standard', standard_pipeline, standard_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols)
        ])
        
        return preprocessor, X_train, X_test