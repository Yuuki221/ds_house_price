import pandas as pd 

class MSSubclassFeatures:
    '''
        MSSubclass features exploration 
    '''

    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def label_mssubclass(self, train):
        '''
            Re-label MSSubclass 
            'MSSubclass' : {
                '20': '1 story 1946+',
                '30': '1 sotry 1945-',
                '40': '1 story unf attic',
                '45': '1,5 story unf',
                '50': '1,5 story fin',
                '60': '2 story 1946+',
                '70': '2 story 1945-',
                '75': '2,5 story all ages',
                '80': 'split/multi level',
                '85': 'split foyer',
                '90': 'duplex all style/age',
                '120': '1 story PUD 1946+',
                '150': '1,5 story PUD all',
                '160': '2 story PUD 1946+',
                '180': 'PUD multilevel',
                '190': '2 family conversion'
            }
        '''

        MSSubclass_Mapping = {
            '20': '1 story 1946+',
            '30': '1 sotry 1945-',
            '40': '1 story unf attic',
            '45': '1,5 story unf',
            '50': '1,5 story fin',
            '60': '2 story 1946+',
            '70': '2 story 1945-',
            '75': '2,5 story all ages',
            '80': 'split/multi level',
            '85': 'split foyer',
            '90': 'duplex all style/age',
            '120': '1 story PUD 1946+',
            '150': '1,5 story PUD all',
            '160': '2 story PUD 1946+',
            '180': 'PUD multilevel',
            '190': '2 family conversion' 
        }

        feature_df = train[['MSSubClass']]
        train['MSSubClass'] = feature_df.fillna('NaN').applymap(lambda x: MSSubclass_Mapping[x])

