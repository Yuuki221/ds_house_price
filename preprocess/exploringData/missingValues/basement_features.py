import pandas as pd 
import seaborn as sns

class BasementFeatures:
    '''
        Basement related features exploration 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')
    
    def relabel_bsmtqual(self, train):
        '''
            relabel the basement quality 
        '''
        Basement_Quality = {
            'NaN': 0,
            'Po': 1, 
            'Fa': 2, 
            'TA': 3, 
            'Gd': 4, 
            'Ex': 5
        }

        feature_df = train[['BsmtQual']]
        train['BsmtQual'] = feature_df.fillna('NaN').applymap(lambda x: Basement_Quality[x])

    def relabel_bsmtcond(self, train):
        '''
            relabel the basement condition variable 
        '''
        Basement_Condition = {
            'NaN': 0,
            'Po': 1, 
            'Fa': 2, 
            'TA': 3, 
            'Gd': 4, 
            'Ex': 5
        }

        feature_df = train[['BsmtCond']]
        train['BsmtCond'] = feature_df.fillna('NaN').applymap(lambda x: Basement_Condition[x])

    def relabel_bsmtExposure(self, train):
        '''
            relabel the basement exposure variable 
        '''

        Basement_Exposure = {
            'NaN': 0, 
            'No': 1, 
            'Mn': 2, 
            'Av': 3,
            'Gd': 4
        }

        feature_df = train[['BsmtExposure']]
        train['BsmtExposure'] = feature_df.fillna('NaN').applymap(lambda x: Basement_Exposure[x])

    def relabel_bsmtFinType1(self, train):
        '''
            relabel bsmtFinType1 variable
        '''
        Basement_Fin_Type1 = {
            'NaN': 0, 
            'Unf': 1, 
            'LwQ': 2, 
            'Rec': 3, 
            'BLQ': 4, 
            'ALQ': 5, 
            'GLQ': 6
        }

        feature_df = train[['BsmtFinType1']]
        train['BsmtFinType1'] = feature_df.fillna('NaN').applymap(lambda x: Basement_Fin_Type1[x])

    def relabel_bsmtFinType2(self, train):
        '''
            relabel bsmtFinType2 variable 
        '''
        Basement_Fin_Type2 = {
            'NaN': 0, 
            'Unf': 1, 
            'LwQ': 2, 
            'Rec': 3, 
            'BLQ': 4, 
            'ALQ': 5, 
            'GLQ': 6
        }

        feature_df = train[['BsmtFinType2']]
        train['BsmtFinType2'] = feature_df.fillna('NaN').applymap(lambda x: Basement_Fin_Type2[x])

    