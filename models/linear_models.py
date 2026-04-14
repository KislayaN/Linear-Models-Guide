from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,  RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

import numpy as np

import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

ALPHAS = np.logspace(-5, 3, 50)

class Runner:
    def __init__(self):
        self.models = {
            'Linear': LinearRegression(),
            'Ridge': RidgeCV(alphas=ALPHAS),
            'Lasso': LassoCV(alphas=ALPHAS, max_iter=10000),
            'ElasticNet': ElasticNetCV(alphas=ALPHAS, l1_ratio=[0.1, 0.5, 0.9], max_iter=10000),
        }
        self.results = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_names):

        for name, model in self.models.items():

            model_instance = clone(model)
            model_instance.fit(X_train, y_train)

            y_pred = model_instance.predict(X_test)

            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            result = {
                "Test_RMSE": rmse,
                "Test_R2": r2
            }
            
            if hasattr(model_instance, "alpha_"):
                result["Best Alpha"] = model_instance.alpha_

            if hasattr(model_instance, "l1_ratio_"):
                result["Best L1 Ratio"] = model_instance.l1_ratio_

            if hasattr(model_instance, "coef_") and len(feature_names) == len(model_instance.coef_):
                result["Coefficients"] = dict(zip(feature_names, model_instance.coef_))

            self.results[name] = result

        return self.results