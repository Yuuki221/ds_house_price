import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

class NumericalVarsImportance:
    def __init__(self):
        self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')
        self.numerical_df = self.train.select_dtypes(include=['float64', 'int'])

    def plot_correlation_graph(self):
        '''
            get correlation graph 
        '''
        corrmat = self.numerical_df.corr()
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 16)
        sns.heatmap(corrmat)

    def correlation(self, threshold):
        '''
            get correlation table 
        '''
        col_corr = set()
        corr_matrix = self.numerical_df.corr().filter(['SalePrice']).drop(['SalePrice'])
        for index, row in corr_matrix.iterrows():
            if abs(row['SalePrice']) > threshold: 
                col_corr.add(index)
        return col_corr
    
    def get_correlated_features(self, corr_rate):
        '''
            get correlated features 
        '''
        return self.correlation(self, corr_rate)
