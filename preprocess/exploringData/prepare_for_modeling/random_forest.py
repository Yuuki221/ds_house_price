import pandas as pd # data processing, CSV file I/O 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel 
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score

import categorical_features
import get_preprocessed_data

class FeatureSelectionRandomForest:
    def __init__(self):
        self.preprocessed = pd.read_csv('~/ds_house/data/preprocess_2.csv', encoding='utf-8')
    
    def correlation(self, dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: 
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr

    def get_correlated_features(self, correlation_data, corr_rate):
        return self.correlation(self, correlation_data, corr_rate)