# from sklearn.datasets import fetch_california_housing

import pandas as pd

dataframe = pd.read_csv("C:/Users/kisla/OneDrive/Desktop/AmesHousing.csv")

class Load_data:
    def __init__(self):
        self.X_dataset = None
        self.y_dataset = None
        self.dataframe = None
        
    def load(self):
        self.dataframe = dataframe
        
        self.y_dataset = self.dataframe['SalePrice']
        self.X_dataset = self.dataframe.drop(columns=["SalePrice"])
        
        return self.X_dataset, self.y_dataset
    
dafa = Load_data()
dafa.load()