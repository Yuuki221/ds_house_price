import pandas as pd 
import seaborn as sns 

class Fireplace:
    '''
        Fireplace exploration 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def check_fireplace_data(self, train):
        fireplaceQuNaNNum = train['FireplacesQu'].isna().sum()
        fireplaceZeroNum = train['Fireplaces'].isin([0]).sum()

        return [fireplaceQuNaNNum, fireplaceZeroNum]

    def label_fireplace_quality(self, train):
        '''
            Re-label fireplace quality 
            'FireplaceQu': {
                'NaN': 0,
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            }
        '''
        FireplaceQu_Mapping = {
            'NaN': 0, 
            'Po': 1,
            'Fa': 2,
            'TA': 3,
            'Gd': 4,
            'Ex': 5
        }

        feature_df = train[['FireplaceQu']]
        train['FireplaceQu'] = feature_df.fillna('NaN').applymap(lambda x: FireplaceQu_Mapping[x])

    
