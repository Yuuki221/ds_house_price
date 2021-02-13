import pandas as pd 

class HeatingAndAircondition:
    '''
        Heating and Aircondition exploration 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def label_heatingqc(self, train):
        '''
            Re-label HeatingQC 
            'HeatingQC': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            }
        '''
        HeatingQC_Mapping = {
            'NaN': 0, 
            'Po': 1,
            'Fa': 2,
            'TA': 3,
            'Gd': 4,
            'Ex': 5
        }

        feature_df = train[['HeatingQC']]
        train['HeatingQC'] = feature_df.fillna('NaN').applymap(lambda x: HeatingQC_Mapping[x])


    def label_CentralAir(self, train):
        '''
            Re-label CentralAir
        '''
        CentralAir_Mapping = {
            'NaN': 0,
            'Y': 1,
            'N': 0
        }

        feature_df = train[['CentralAir']]
        train['CentralAir'] = feature_df.fillna('NaN').applymap(lambda x: CentralAir_Mapping[x])

    