import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Neighborhoods:
    '''
        plot neighborhood and group by their groups 
    '''
    def __init__(self):
        self.train = pd.read_csv('~/ds_house/preprocess_5.csv', encoding='utf-8')

    def plot_neighbor_median(self, dataset):
        '''
            plot neighborhood and group by their groups 
        '''
        neighbor_price = dataset[['Neighborhood', 'SalePrice']]
        price_groupby_neighbor = neighbor_price.groupby(by='Neighborhood')
        mean_price_neighbor = price_groupby_neighbor['SalePrice'].mean()
        mean_dataframe = mean_price_neighbor.to_frame().sort_values(by='SalePrice')
        fig, ax = plt.subplots()
        x = np.arange(25)
        plt.bar(x, mean_dataframe['SalePrice'])
        plt.xticks(x, 
                  ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'],
                  fontsize=9, 
                  rotation=90)
        plt.show()

    
    def neighbor_relabel(self, row):
        '''
            re-group neighborhood into three groups 
        '''
        group_1 = {'StoneBr', 'NridgHt', 'NoRidge'}
        group_2 = {'MeadowV', 'IDOTRR', 'BrDale'}
        if row['Neighborhood'] in group_1:
            return 2 
        elif row['Neighborhood'] in group_2:
            return 0
        else:
            return 1
    
    def factor_neighbors(self):
        '''
            Refactor neighbors based on the frequency 
        '''
        dataset = self.train

        if 'NeighborType' not in dataset:
            dataset['NeighborType'] = dataset.apply(lambda row: self.neighbor_relabel(row), axis=1)

        return dataset.to_csv('preprocess_6.csv')

neighborhoods = Neighborhoods()
neighborhoods.factor_neighbors()



