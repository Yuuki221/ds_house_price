import pandas as pd 

class Functional:
    '''
        Functional exploration 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def relabel_functional(self, train):
        '''
            Relabel functional based on ordinal
        '''
        Functional_Mapping = {
            'NaN': 0,
            'Sal': 0,
            'Sev': 1, 
            'Maj2': 2,
            'Maj1': 3, 
            'Mod': 4,
            'Min2': 5, 
            'Min1': 6,
            'Typ': 7
        }

        feature_df = train[['Functional']]
        train['Functional'] = feature_df.fillna('NaN').applymap(lambda x: Functional_Mapping[x])