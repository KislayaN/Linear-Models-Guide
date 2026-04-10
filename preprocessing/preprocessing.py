import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

class Preprocessing:
    def __init__(self, plan=None, X_train=None, X_test=None, y_train=None):
        self.plan = plan
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        
        self.columns_to_drop = plan["Drop Corr"] + plan["Drop ID"]
        self.vif_selected_cols = plan["VIF Selected"]
        self.log_transform_cols = plan["Log Transform"]
        self.robust_scale_cols = plan["Robust Scale"]
        self.standard_scale_cols = plan["Standard Scale"]
        self.ohe_encode_cols = plan["OHE (One Hot Encoding)"]
        self.high_card_encode = plan["High Card Encode"]
        
    def get_categoric_cols(self, dataframe):
        return dataframe.select_dtypes(include=['object', 'bool']).columns.to_list()
    
    def get_numeric_cols(self, dataframe):
        return dataframe.select_dtypes(include=['float', 'int']).columns.to_list()
    
    def target_encode(self, X_train=None, y_train=None, high_card_cols=None, X_test=None):
        global_mean = y_train.mean()
        X_train = X_train.copy()
        X_test = X_test.copy()
    
        for col in high_card_cols:
            mapping = pd.concat([X_train[col], y_train], axis=1).groupby(col)[y_train.name].mean()
            
            if X_train is not None:
                X_train[col] = X_train[col].map(mapping)
                
            if X_test is not None:
                X_test[col] = X_test[col].map(mapping)
            
                # handle unseen categories
                X_test[col] = X_test[col].fillna(global_mean)
        
        return X_train, X_test
        
    def preprocess(self):
        
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        
        y_train = self.y_train.copy()
        
        to_drop = [c for c in self.columns_to_drop if c in X_train.columns]
        
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
        
        valid_vif_cols = [c for c in self.vif_selected_cols if c in X_train.columns]
        
        X_train = X_train[valid_vif_cols]
        X_test = X_test[valid_vif_cols]
        
        high_card_encode = [c for c in self.high_card_encode if c in X_train.columns]
        
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
        
        self.execute = ColumnTransformer([
            ('log', log_pipeline, log_cols),
            ('robust', robust_pipeline, robust_cols),
            ('standard', standard_pipeline, standard_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols)
        ])
        
        return X_train, X_test