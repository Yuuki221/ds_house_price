import pandas as pd 
import seaborn as sns 

class LotFeatures:
    '''
        Lot related features 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def replace_lotFrontage_with_median(self, train):
        '''
            Replace NaN value in lotFrontage with median of each neighborhood
        '''
        lotValue_neighborhood_features = ['LotConfig', 'Neighborhood', 'LotShape', 'LotFrontage']
        lot_data = train[lotValue_neighborhood_features]
        lot_Frontage = lot_data[['LotFrontage', 'Neighborhood']]
        lotFrontage_median = lot_Frontage.groupby(by=['Neighborhood']).median()

        # create a set for lotfrontage_median and neighborhood 
        neighbor_lotFrontage_set = dict()
        for index, row in lotFrontage_median.iterrows():
            neighbor_lotFrontage_set[index] = row['LotFrontage']
        
        # replace training set value with median value of the group 
        for index, row in train.iterrows():
            if pd.isnull(row['LotFrontage']):
                train.at[index, 'LotFrontage'] = neighbor_lotFrontage_set[row['Neighborhood']]


