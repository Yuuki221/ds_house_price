import pandas as pd 

class ExteriorFeatures:
    '''
        Exterior feature exploration 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def relabel_exterQual(self, train):
        '''
            Re-label ExterQual
        '''
        ExterQual_Mapping = {
            'NaN': 0, 
            'Po': 1, 
            'Fa': 2,
            'TA': 3,
            'Gd': 4,
            'Ex': 5
        }

        feature_df = train[['ExterQual']]
        train['ExterQual'] = feature_df.fillna('NaN').applymap(lambda x: ExterQual_Mapping[x])

    def relabel_exterCond(self, train):
        '''
            Re-label ExterCond
        '''
        ExterCond_Mapping = {
           'NaN': 0, 
           'Po': 1, 
           'Fa': 2,
           'TA': 3,
           'Gd': 4,
           'Ex': 5 
        }

        feature_df = train[['ExterCond']]
        train['ExterCond'] = feature_df.fillna('NaN').applymap(lambda x: ExterCond_Mapping[x])
