import pandas as pd 
import seaborn as sns

class Alley:
    '''
        Alley is the next variable to be processed 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def plot_alley(self, train):
        '''
            get alley frequency plot 
        '''
        frequencies = train['Alley'].value_counts(dropna=False)
        frequencies.plot.bar(legend=True, title='Frequency of Alley', grid=True)
    
    