import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class ProcessSalesPrice:
    def __init__(self):
        self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def plot_saleprice_histogram(self):
        '''
            get salesprice histogram 
        '''
        sale_price = self.train['SalePrice']
        plt.hist(sale_price)
        plt.show()
