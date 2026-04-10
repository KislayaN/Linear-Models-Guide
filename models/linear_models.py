from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,  RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

models = {
    'Linear': LinearRegression(),
    'Ridge-CV': RidgeCV(),
    'Ridge': Ridge(),
    'Lasso-CV': LassoCV(),
    'Lasso': Lasso(),
    'EN-CV': ElasticNetCV(),
    'Elastic_Net': ElasticNet(),
}

class Runner:
    def __init__(self):
        self.models = models
        self.results = {}
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.results[name] = {
                "RMSE": rmse,
                "R2": r2
            }
            
        return self.results
    
    def cross_validate(self, X_train, y_train, cv=5):
        
        for name, model in self.models.items():
            
            scores = cross_val_score(
                model, X_train, y_train,
                scoring='neg_root_mean_squared_error',
                cv=cv
            )
            
            self.results[name]["CV_RMSE"] = -scores.mean()
        
        return self.results