import pandas as pd 

class KitchenFeatures:
    '''
        Explore Kitchen related features 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def label_kitchen_quality(self, train):
        '''
            Re-label kitchen quality
            
        '''
        Kitchen_Quality_Mapping = {
            'NaN': 0, 
            'Po': 1,
            'Fa': 2,
            'TA': 3,
            'Gd': 4,
            'Ex': 5    
        }

        feature_df = train[['KitchenQual']]
        train['KitchenQual'] = feature_df.fillna('NaN').applymap(lambda x: Kitchen_Quality_Mapping[x])