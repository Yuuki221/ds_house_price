import pandas as pd 

class MasoryFeatures: 
    '''
        Masory Features Data Exploration 
    '''

    # def __init__(self):
    #    self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8') 

    def get_mansory_saleprice_median(self, train):
        '''
            get median sale price for each group of mansory type
        '''

        mansoryType_salePrice_median = train[['MasVnrType', 'SalePrice']].groupby(by=['MasVnrType']).median()
        return mansoryType_salePrice_median

    def label_masVnrType(self, train):
        '''
            Re-label MasVnrType based on average saleprice 
        '''

        MasVnrType_Mapping = {
            'NaN': 0, 
            'None': 0, 
            'BrkCmn': 0,
            'BrkFace': 1, 
            'Stone': 2
        }

        feature_df = train[['MasVnrType']]
        train['MasVnrType'] = feature_df.fillna('NaN').applymap(lambda x: MasVnrType_Mapping[x])

    def impute_masVnrArea(self, train):
        '''
            impute NaN values to 0s 
        '''
        train['MasVnrArea'] = train['MasVnrArea'].fillna(0.0)