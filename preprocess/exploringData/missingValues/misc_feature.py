import pandas as pd 
import seaborn as sns

class MiscFeature:
    '''
        MiscFeatures is the fixing 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def plot_misc_features(self, train):
        '''
            Get the plot for MiscFeatures 
        '''
        frequencies = train['MiscFeatures'].value_counts(dropna=False)
        frequencies.plot.bar(legend=True, title='Frequency of MiscFeatures', grid=True)