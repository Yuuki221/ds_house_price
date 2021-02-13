import pandas as pd 

class PoolQuality:
    '''
        Pool Quality is the variable that has most Na values 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')
    
    def plot_pool_quality(self, train):
        '''
            get pool quality frequency plot
        '''
        frequencies = train['PoolQC'].value_counts(dropna=False)
        frequencies.plot.bar(legend=True, title='Frequency of Pool Quality', grid=True)

    def label_pool_quality(self, train):
        '''
            Re-label pool quality 
            'PoolQC': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            }
        '''
        PoolQC_Mapping = {
            'NaN': 0, 
            'Po': 1, 
            'Fa': 2,
            'TA': 3,
            'Gd': 4,
            'Ex': 5
        }

        feature_df = train[['PoolQC']]
        train['PoolQC'] = feature_df.fillna('NaN').applymap(lambda x: PoolQC_Mapping[x])

    