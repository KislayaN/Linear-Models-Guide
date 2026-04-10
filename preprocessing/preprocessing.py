import sys
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.get_data import Load_data
from eda.eda import EDA

data_loader = Load_data()
data = data_loader.load()

eda_instance = EDA(data=data)
skewness_stats = eda_instance.get_skewed_cols()
outliers_stats_df = eda_instance.get_outliers()
feature_names = eda_instance.feature_columns
corr_pairs = eda_instance.correlation_matrix()

class Preprocessing:
    def __init__(self, dataframe, feature_names):
        self.dataframe = dataframe
        self.feature_names = feature_names
        self.outliers_flag = {}
        
    def get_categoric_cols(self, dataframe):
        return dataframe.select_dtypes(include=['object', 'bool']).columns.to_list()
    
    def get_numeric_cols(self, dataframe):
        return dataframe.select_dtypes(include=['float', 'int']).columns.to_list()
    
    def has_extreme_outliers(self, dataframe):
        for col in self.feature_names:
            p99 = dataframe[col].quantile(0.99)
            
            Q3 = dataframe[col].quantile(0.75)
            Q1 = dataframe[col].quantile(0.25)
            
            IQR = Q3 - Q1
            
            if (p99 - Q3) / IQR > 3:
                print("Robust scaler")
            else: 
                print("Standard Scaler") 
        
    def preprocess(self):
        X_dataframe = self.dataframe[0].copy()
        
        for i in range(outliers_stats_df.shape[0]):
            value = outliers_stats_df['outliers_percentage_per_col'].iloc[i]
            
            if value >= 0.2:
                self.outliers_flag[outliers_stats_df.index[i]] = value
            
        # Getting categoric columns
        self.categoric_cols = self.get_categoric_cols(dataframe=X_dataframe)
        
        # Getting numeric columnns
        self.numeric_cols = self.get_numeric_cols(dataframe=X_dataframe)
        
        numeric_pipeline = Pipeline({
            ('log', FunctionTransformer(np.log1p, validate=False)),
            ('log', StandardScaler())
        })
        
        categoric_pipeline = Pipeline([
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            ('num', numeric_pipeline, self.numeric_cols),
            ('num', categoric_pipeline, self.categoric_cols)
        )
        