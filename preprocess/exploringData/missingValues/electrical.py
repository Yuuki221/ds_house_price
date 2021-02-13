import pandas as pd 

class Electrical:
    '''
        Electrical exploration 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def impute_eletrical_value(self, train):
        '''
            Impute electrical value by mode 
        '''
        for index, row in train.iterrows():
            if pd.isnull(row['Electrical']):
                train.at[index, 'Electrical'] = train['Electrical'].mode()