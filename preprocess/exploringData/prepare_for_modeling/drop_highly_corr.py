import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

class DropHighlyCorr:
    '''
        remove highly correlated variables 
    '''
    def remove_correlated_variables(self, dataset, varsname):
        '''
            remove highly related variables 
        '''
        dataset.drop(varsname, axis=1)

    def remove_outliers(self, dataset, outliers):
        '''
            remove outliers 
        '''
        outliers_index = []
        
        for i, row in post_train.iterrows():
            if row['Id'] in outliers:
                outliers_index.append(i)

        dataset = dataset.drop(outliers_index, axis=0)