import pandas as pd 

class LandFeatures:
    '''
        Explore Land Features variable 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def label_landSlope(self, train):
        '''
            Re-label land slope 
            'LandSlope': {
                'NaN': 0,
                'Sev': 0,
                'Mod': 1,
                'Gtl': 2
            }
        '''
        LandSlope_Mapping = {
            'NaN': 0,
            'Sev': 0,
            'Mod': 1,
            'Gtl': 2
        }

        feature_df = train[['LandSlope']]
        train['LandSlope'] = feature_df.fillna('NaN').applymap(lambda x: LandSlope_Mapping[x])

    
    