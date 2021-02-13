import pandas as pd 
import numpy as np 

class CategoricalFeatures:
    '''
        get categorical features 
    '''
    def get_categorical_variables(self):
        '''
            get categorical variables 
        '''
        return [
            'MSSubClass',
            'OverallQual',
            'OverallCond',
            'MasVnrType',
            'ExterQual',
            'ExterCond',
            'BsmtQual',
            'BsmtCond',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtFinType2',
            'KitchenQual',
            'Functional',
            'FireplaceQu',
            'GarageCond',
            'GarageQual',
            'GarageFinish',
            'PoolQC',
            'Remodel',
            'NeighborType'
        ]

    def get_object_categorical(self):
        '''
            Get categorical data 
        '''
        return [
            'MSZoning',
            'Street',
            'Alley',
            'LotShape',
            'LandContour',
            'Utilities',
            'LotConfig',
            'LandSlope',
            'Condition1',
            'Condition2',
            'BldgType',
            'HouseStyle',
            'RoofStyle',
            'RoofMatl',
            'Exterior1st',
            'Exterior2nd',
            'Foundation',
            'Heating',
            'HeatingQC',
            'CentralAir',
            'Electrical',
            'GarageType',
            'PavedDrive',
            'Fence',
            'MiscFeature',
            'SaleType',
            'SaleCondition'
        ]
        