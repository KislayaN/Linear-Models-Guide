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
        self.corr_pairs = None
        
    
    def analyze_skew(self):
        # Select only numerical columns
        num_cols = self.dataframe_X.select_dtypes(include=['int64', 'float64']).columns
        
        skewness = {}
        
        for col in num_cols:
            skew_value = self.dataframe_X[col].skew()
            
            # classify skew
            if abs(skew_value) > 1:
                level = "Highly Skewed"
            elif abs(skew_value) > 0.5:
                level = "Moderately Skewed"
            else:
                level = "Normal"
            
            skewness[col] = {
                "skew_value": round(skew_value, 3),
                "type": level
            }
        
        self.skewness_stats = pd.DataFrame(skewness).T
        
        return self.skewness_stats
    
    def analyze_outliers(self):
        results = {}
        
        num_cols = self.dataframe_X.select_dtypes(include=['int64', 'float64']).columns
        
        for col in num_cols:
            series = self.dataframe_X[col].dropna()
            
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
                    "level": level
                }
    
        self.outliers_stats_df = pd.DataFrame(results).T
    
        return self.outliers_stats_df
    
    def analyze_corr(self, threshold=0.8):
        # Used for detecting linear relationships in the dataset 
        
        num_df = self.dataframe_X.select_dtypes(include=['int64', 'float64'])
        
        corr_matrix = num_df.corr().abs()
        
        mask = np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool)
        
        upper = corr_matrix.where(mask)
        
        pairs = upper.stack().reset_index()
        pairs.columns = ['feature-1','feature-2', 'correlation']
        
        pairs = pairs[pairs['correlation'] > threshold]
        pairs = pairs.sort_values(by='correlation', ascending=False)
        
        self.corr_pairs = pairs 
        
        return self.corr_pairs
    
    def features_to_drop(self, threshold=0.8):
        X_dataframe = self.dataframe_X.select_dtypes(include=['int64','float64'])
        
        corr_matrix = X_dataframe.corr().abs()
        target_corr = X_dataframe.corrwith(self.dataframe_y).abs()
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        
        for col in upper.columns:
            for row in upper.index:
                if upper.loc[row, col] > threshold:
                    
                    # compare with target
                    
                    if target_corr[row] < target_corr[col]:
                        to_drop.add(row)
                    else: 
                        to_drop.add(col)
        
        return list(to_drop)
    
    def analyze_cardinality(self):
        self.categoric_cols = self.dataframe_X.select_dtypes(include=['bool', 'object']).columns.tolist()
        
        results = {}
        
        if self.categoric_cols:
            for col in self.categoric_cols:
                unique_count = self.dataframe_X[col].nunique()
                ratio = unique_count / len(self.dataframe_X)
                
                if ratio > 0.9:
                    level = "ID like"
                    suggestion = "Drop these column"
                
                elif ratio > 0.2:
                    level = "Very High Cardinality"
                    suggestion = "Use target/Frequency encoding"
                    
                elif ratio > 0.1:
                    level = "High Cardinality"
                    suggestion = "Group Rare Categories"
                    
                elif ratio > 0.01:
                    level = "Moderate Cardinality"
                    suggestion = "OHE"
                    
                else: 
                    level = "Low Cardinality"
                    suggestion = "OHE"
                    
                results[col] = {
                    "Unique": unique_count,
                    "Ratio": round(ratio, 3),
                    "Cardinality": level,
                    "Suggestion": suggestion
                }
            
            card_df = pd.DataFrame(results).T
            
            id_cols = card_df[card_df['Ratio'] > 0.9].index.tolist()

            low_card_cols = card_df[card_df['Ratio'] <= 0.1].index.tolist()

            high_card_cols = card_df[
                (card_df['Ratio'] > 0.1) & (card_df['Ratio'] <= 0.9)
            ].index.tolist()
        
        if not self.categoric_cols:
            return [], [], []
            
        return low_card_cols, high_card_cols, id_cols
        
    def analyze_vif(self, threshold = 10):
        
        X_dataframe = self.dataframe_X.copy()
        
        X_dataframe = X_dataframe.select_dtypes(include=['int64', 'float64'])
        X_dataframe = X_dataframe.fillna(X_dataframe.median())
        
        dropped_cols = []
        
        while True:
            vif_data = pd.DataFrame()
            vif_data['Features'] = X_dataframe.columns
            
            vif_data['VIF'] = [
                variance_inflation_factor(X_dataframe.values, i)
                for i in range(X_dataframe.shape[1])
            ]
            
            max_vif = vif_data["VIF"].max()
            
            if max_vif < threshold:
                break
            
            drop_feature = vif_data.sort_values(by='VIF', ascending=False)['Features'].iloc[0]
            dropped_cols.append(drop_feature)
            
            X_dataframe = X_dataframe.drop(columns=[drop_feature])
            
        self.vif_result = vif_data.sort_values(by='VIF', ascending=False)
        self.vif_selected_features = X_dataframe.columns.tolist()
        self.vif_dropped_features = dropped_cols
            
        return self.vif_result, self.vif_selected_features
    
    def get_skewed_cols(self):
        skew_df = self.analyze_skew()
        
        return skew_df[skew_df['type'] == "Highly Skewed"].index.tolist()
    
    def get_outlier_cols(self):
        outliers_df = self.analyze_outliers()
        
        return outliers_df[outliers_df['level'].isin(['High', 'Very High'])].index.tolist()
    
    def build_preprocessing_plan(self):
        lowcard, highcard, idcard = self.analyze_cardinality()
        _, vif_cols = self.analyze_vif()
        
        skew_cols = self.get_skewed_cols()
        log_cols = skew_cols
        
        outlier_cols = self.get_outlier_cols()
        robust_cols  =[col for col in outlier_cols if col not in log_cols]
        return {
            "Log Transform": log_cols,
            "Robust Scale": robust_cols,
            "Standard Scale": [
                col for col in self.dataframe_X.select_dtypes(include=['int64', 'float64']).columns
                if col not in log_cols + robust_cols 
            ],
            "OHE (One Hot Encoding)": lowcard,
            "High Card Encode": highcard,
            "Drop Corr": self.features_to_drop(),
            "Drop ID": idcard,
            "VIF Selected": vif_cols 
        }