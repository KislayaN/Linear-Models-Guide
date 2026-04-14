import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor

class EDA:
    def __init__(self, data):
        self.data = data
        self.dataframe_X = data[0]
        self.dataframe_y = data[1]
        self.feature_columns = data[0].columns
        self.skewness_stats = None
        self.outliers_stats_df = None
    
    def analyze_skew(self):
        # Select only numerical columns
        num_df = self.dataframe_X.select_dtypes(include=[np.number])
        
        skew_series = num_df.skew()
        
        skew_df = skew_series.to_frame(name='skew_value')
        skew_df = skew_df.dropna()
            
        def classify(val):
            # classify skew
            if abs(val) > 1:
                return "Highly Skewed"
            elif abs(val) > 0.5:
                return "Moderately Skewed"
            else:
                return "Normal"
            
        skew_df['type'] = skew_df['skew_value'].apply(classify)
        
        skew_df['skew_value'] = skew_df['skew_value'].round(3)
        
        self.skewness_stats = skew_df
        return skew_df
    
    def analyze_outliers(self):
        results = {}
        
        num_df = self.dataframe_X.select_dtypes(include=[np.number])
        
        for col in num_df.columns:
            series = self.dataframe_X[col].dropna()
            
            if len(series) == 0:
                continue
            
            Q3 = series.quantile(0.75)
            Q1 = series.quantile(0.25)
            IQR = Q3- Q1
            
            upper = Q3 + 1.5 * IQR
            lower = Q1 - 1.5 * IQR
            
            outliers = ((series < lower) | (series > upper)).sum()
            percentage = (outliers / len(series)) * 100
            
            if percentage > 20:
                level = "Very High"
            elif percentage > 10:
                level = "High"
            elif percentage > 5:
                level = "Moderate"
            else:
                level = "Low"
        
            results[col] = {
                    "outlier_%": round(percentage, 2),
                    "level": level,
                    "Upper_boundary": upper,
                    "Lower_boundary": lower
                }
    
        outliers_df = pd.DataFrame(results).T
        self.outliers_stats_df = outliers_df
    
        return outliers_df
    
    def analyze_cardinality(self):
        cat_cols = self.dataframe_X.select_dtypes(
            include=['bool', 'object', 'category']
        ).columns.tolist()
        
        results = {}
        
        for col in cat_cols:
            series = self.dataframe_X[col]
            
            unique_count = series.nunique(dropna=True)
            total = len(series)
            ratio = unique_count / len(series) if total > 0 else 0
                
            if ratio > 0.9 and unique_count > 50:
                level = "ID like"
            elif unique_count > 30:
                level = "Very High Cardinality"
            elif unique_count > 10:
                level = "High Cardinality" 
            elif unique_count > 3:
                level = "Moderate Cardinality"
            else: 
                level = "Low Cardinality"
               
            results[col] = {
                    "Unique": unique_count,
                    "Ratio": round(ratio, 3),
                    "Cardinality": level,
                }
            
        card_df = pd.DataFrame(results).T if results else pd.DataFrame()
        
        self.cardinality_stats = card_df
        
        return card_df
    
    def get_skewed_cols(self):
        skew_df = self.analyze_skew()
        
        return skew_df[skew_df['type'] == "Highly Skewed"].index.tolist()
    
    def get_outlier_cols(self):
        outliers_df = self.analyze_outliers()
        
        return outliers_df[outliers_df['level'].isin(['High', 'Very High'])].index.tolist()
    
    def build_preprocessing_plan(self):
        card_df = self.analyze_cardinality()
        skew_cols = self.get_skewed_cols()
        outlier_cols = self.get_outlier_cols()
        
        return {
            "skewed_cols": skew_cols,
            "outlier_cols": outlier_cols,
            "low_card_cols": card_df[card_df["Cardinality"] == "Low Cardinality"],
            "high_card_cols": card_df[
                card_df["Cardinality"].isin([
                    'Moderate Cardinality',
                    'High Cardinality',
                    'Very High Cardinality'
                ])
                ].index.tolist(),
            "id_cols": card_df[card_df["Cardinality"] == "ID Like"]
        }