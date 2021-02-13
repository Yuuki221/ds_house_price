import pandas as pd 
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt 

class NormalizeFeatures:
    
    '''
        observe the distribution of variables 
        and normalize the distribution 

    '''
    
    def get_original_plot(self, feature, dataset):
        '''
            get the original plot of certain feature 

        '''
        plt.style.use('ggplot')
        plt.hist(dataset[feature], bins=60)

    def get_normalized_plot(self, feature, dataset):
        '''
            get the normalized plot of certain feature 

        '''
        plt.style.use('ggplot')
        var_data = dataset[[feature]]
        data = var_data.applymap(lambda x: np.log(x+2))
        plt.hist(data[feature], bins=60)
    

    def normailize_feature(self, feature, dataset):
        '''
            normalized the feature based on the plots 
        '''
        dataset[feature] = dataset[[feature]].applymap(lambda x: np.log(x+2))   