import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

class SquareFeet:
    '''
        process square feet related features 
        GrLivArea TotalBsmtSF
        Total living space generally is very important 
        when people buying houses, adding a predictors that adds up the living spaces 
    '''
    def create_totalSqFeet(self, dataset):
        '''
            add GrLivArea and TotalBsmtSF to TotalLivingSpace
        '''
        if 'TotalArea' not in dataset:
            dataset['TotalArea'] = dataset.apply(lambda row: self.add_GrLivArea_TotalBsmtSF(row), axis=1)
    
    def add_GrLivArea_TotalBsmtSF(self, row):
        '''
            add GrLivArea and TotalBsmtSF 
        '''
        return row['GrLivArea'] + row['TotalBsmtSF']

    def plot_scatter_totalSqFeet(self, dataset):
        '''
            plot scatter plot for totalSqFeet to check for outliers 
        '''
        sns.set_theme(color_codes=True)
        sns.regplot(x='GrLivArea', y='SalePrice', data=dataset)
        plt.title('Scatter plot of SalePrice and GrLivArea')
        self.label_point(dataset['TotalBsmtSF'], dataset['SalePrice'], dataset['Id'], plt.gca())

    def label_point(self, x, y, val, ax):
        '''
            label the point with row Id, so that we could identify outlier
        '''
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(point['val']))


        