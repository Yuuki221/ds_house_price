import pandas as pd 
import seaborn as sns

class Fence:
    '''
        Fence variable exploration 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')
    
    def plot_pool_quality(self, train):
        '''
            get pool quality frequency plot
        '''
        frequencies = train['Fence'].value_counts(dropna=False)
        frequencies.plot.bar(legend=True, title='Frequency of Fence', grid=True)
