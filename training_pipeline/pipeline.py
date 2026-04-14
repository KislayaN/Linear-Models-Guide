from sklearn.model_selection import train_test_split

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.get_data import Load_data
from eda.eda import EDA
from preprocessing.preprocessing import Preprocessing
from models.linear_models import Runner

import numpy as np
import pandas as pd
        
class Training_pipeline:
    def __init__(self):
        self.dataframe = None
        
    def get_top_features(self, results, model_name, top_n=10):
        coefs = results[model_name]["Coefficients"]
        
        df = pd.DataFrame({
            "feature": coefs.keys(),
            "coef": coefs.values()
        })
        
        df["feature"] = df["feature"].apply(lambda x: x.split("__")[-1])
        df["abs_coef"] = df["coef"].abs()
        
        df = df.sort_values(by="abs_coef", ascending=False)
        
        return df.head(top_n)[["feature", "coef"]]
    
    def run(self):
        # Get the data 
        data_loader = Load_data()
        self.dataframe = data_loader.load()
        
        # EDA
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataframe[0],
            self.dataframe[1],
            test_size=0.3,
            random_state=42)
        
        eda = EDA(data=(X_train, y_train))
        plan = eda.build_preprocessing_plan()
        
        
        # Preprocessing 
        preprocessing = Preprocessing(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            plan=plan
        )
        
        X_train_cleaned, X_test_cleaned = preprocessing.prepare_data()
        
        X_train_final = preprocessing.column_transformer.fit_transform(X_train_cleaned)
        feature_names = preprocessing.column_transformer.get_feature_names_out()
        X_test_final = preprocessing.column_transformer.transform(X_test_cleaned)
        
        X_train_final = pd.DataFrame(
            X_train_final,
            columns=feature_names,
            index=X_train_cleaned.index
        )
        X_test_final = pd.DataFrame(
            X_test_final,
            columns=feature_names,
            index=X_test_cleaned.index
        )
        
        # Modelling
        runner = Runner()
        
        results = runner.train_and_evaluate(
            X_train=X_train_final,
            X_test=X_test_final,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names
        )
        
        summary = []

        for model, res in results.items():
            row = {
                "Model": model,
                "RMSE": res["Test_RMSE"],
                "R2": res["Test_R2"]
            }
            
            if "Best Alpha" in res:
                row["Alpha"] = res["Best Alpha"]
                
            if "Best L1 Ratio" in res:
                row["L1 Ratio"] = res["Best L1 Ratio"]
            
            summary.append(row)

        summary_df = pd.DataFrame(summary).sort_values(by="R2", ascending=False)

        print("\n=== Model Performance ===")
        print(summary_df.to_string(index=False))

        print("\n=== Top Features (Lasso) ===")
        print(self.get_top_features(results, "Lasso").to_string(index=False))
        
pipe = Training_pipeline()
pipe.run()