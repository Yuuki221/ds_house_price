import pandas as pd
import numpy as np 

class NormalizeSalePrice:
    '''
        Normalize SalePrice
    '''
    def normalize_saleprice(self, dataset):
        dataset['SalePrice'] = dataset[['SalePrice']].applymap(lambda x: np.log(x))
        
