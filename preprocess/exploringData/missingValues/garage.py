import pandas as pd 

class GarageData:
    '''
        Exploring Garage Data related features 
    '''
    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def get_nan_values(self, train):
        '''
            Get the NaN values for Garage related features 
        '''
        garageYrBltNAs = train['GarageYrBlt'].isna().sum()
        garageCarsNAs = train['GarageCars'].isna().sum()
        garageAreaNAs = train['GarageArea'].isna().sum()
        garageCondNAs = train['GarageCond'].isna().sum()
        garageQualNAs = train['GarageQual'].isna().sum()

        return {
            'GarageYrBlt': garageYrBltNAs,
            'GarageCars': garageCarsNAs,
            'GarageArea': garageAreaNAs,
            'GarageCond': garageCondNAs,
            'GarageQual': garageQualNAs
        }

    def impute_GarageYrBlt_with_YearBuilt(self, train):
        '''
            Impute GarageYrBlt with YearBuilt
        '''
        for index, row in train.iterrows():
            if pd.isnull(row['GarageYrBlt']):
                train.at[index, 'GarageYrBlt'] = train.at[index, 'YearBuilt']

    
    def impute_other_garage_with_mode(self, train):
        '''
            impute other garage related features with its mode 
        '''
        garage_data = train[['GarageCars', 'GarageArea', 'GarageCond', 'GarageQual', 'GarageFinish']]
        garage_data_mode = garage_data.mode(dropna=True)

        # GarageCond
        for index, row in train.iterrows():
            if pd.isnull(row['GarageCond']):
                train.at[index, 'GarageCond'] = garage_data_mode.at[0, 'GarageCond']
            if pd.isnull(row['GarageQual']):
                train.at[index, 'GarageQual'] = garage_data_mode.at[0, 'GarageQual']

    def relabel_garageFinish(self, train):
        '''
            relabel garage finish variable 
            GarageFinish:
            {
                'NaN': 0,
                'Unf': 1,
                'Rfn': 2,
                'Fin': 3
            }
        '''
        GarageFinish_Mapping = {
            'NaN': 0, 
            'Unf': 1, 
            'RFn': 2,
            'Fin': 3
        }

        feature_df = train[['GarageFinish']]
        train['GarageFinish'] = feature_df.fillna('NaN').applymap(lambda x: GarageFinish_Mapping[x])
        print(train['GarageFinish'])


    def relabel_garageQual(self, train):
        '''
            relabel garage quality variable 
            GarageQual : {
                'NaN': 0,
                'Po': 1, 
                'Fa': 2,
                'TA': 3, 
                'Gd': 4, 
                'Ex': 5
            }
        '''
        GarageQual_Mapping = {
           'NaN': 0,
           'Po': 1, 
           'Fa': 2,
           'TA': 3, 
           'Gd': 4, 
           'Ex': 5 
        }

        feature_df = train[['GarageQual']]
        train['GarageQual'] = feature_df.fillna('NaN').applymap(lambda x: GarageQual_Mapping[x])

    def relabel_garageCond(self, train):
        '''
            relabel GarageCond variable 
            GarageCond: {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3, 
                'Gd': 4, 
                'Ex': 5
            }
        '''

        GarageCond_Mapping = {
            'NaN': 0, 
            'Po': 1,
            'Fa': 2,
            'TA': 3, 
            'Gd': 4, 
            'Ex': 5 
        }

        feature_df = train[['GarageCond']]
        train['GarageCond'] = feature_df.fillna('NaN').applymap(lambda x: GarageCond_Mapping[x])