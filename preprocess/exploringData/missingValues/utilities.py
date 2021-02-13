import pandas as pd 

class Utilities:
    '''
        Utitlies exploration 
    '''

    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def get_utilities_frequencies(self, train):
        return train['Utilities'].value_counts(dropna=False)

    def remove_utilities(self, train):
        train.drop(columns=['Utilities'])