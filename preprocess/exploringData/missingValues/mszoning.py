import pandas as pd 

class MSZoning:
    '''
        MSZoning variable exploration 
    '''

    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def get_mszoning_frequencies(self, train):
        '''
            get mszoning frequency table 
        '''
        return train['MSZoning'].value_counts(dropna=False)
    