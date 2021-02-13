import pandas as pd 
import numpy as np

class GetNumericalVars:
    '''
        get numerical variables 
    '''
    def get_numerical_variables(self):
        '''
            get numerical variables 
        '''
        return [
            'Age',
            'BedroomAbvGr',
            'BsmtVnrArea',
            'BsmtFinSF1',
            'BsmtFinSF2',
            'BsmtUnfSF',
            'Fireplaces',
            'GarageCars',
            'GarageArea',
            'Kitchen',
            'LowQualFinSF',
            'LotFrontage',
            'LotArea',
            'MasVnrArea',
            'MiscVal',
            'PoolArea',
            'TotalPorch',
            'TotRmsAbvGrd',
            'TotalArea',
            'TotalBathrooms',
            'WoodDeckSF',
            '1stFlrSF',
            '2ndFlrSF'
        ]

    def get_other_numerical_variables(self):
        '''
            Get the numerical variables that are not belong to 
            real numerical variables 
        '''
        return [
            'SalePrice',
            'MoSold',
            'YrSold'
        ]