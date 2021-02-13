import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

class ConsolidatingPorch:
    '''
        consolidate porch 
    '''
    def combine_porch_variables(self, row):
        '''
            combine OpenPorchSF, EnclosedPorch, X3SsnPorch, ScreenPorch
        '''
        return row['OpenPorchSF'] + row['EnclosedPorch'] + row['3SsnPorch'] + row['ScreenPorch']

    def sum_porch_variables(self, dataset):
        '''
            add values of variables about porch 
        '''
        if 'TotalPorch' not in dataset:
            dataset['TotalPorch'] = dataset.apply(lambda row: self.combine_porch_variables(row), axis=1)

    def plot_porch_saleprice(self, dataset):
        '''
            plot total porch sale price scatter plot

        '''
        sns.set_theme(color_codes=True)
        sns.regplot(x='TotalPorch', y='SalePrice', data=dataset)
        plt.title('Scatter plot of SalePrice and TotalPorch')
    
    
        
