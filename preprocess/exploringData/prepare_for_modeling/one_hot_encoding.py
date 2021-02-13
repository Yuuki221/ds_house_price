import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 

class OneHotEncoding:
    '''
        One Hot Encoding for categorical features 
    '''

    def __init__(self):
        self.train = pd.read_csv('~/ds_house/preprocess_4.csv', encoding='utf-8')
    
    def one_hot_encoded(self):
        '''
            one hot encode feature
            create new features and add as one-hot encoded feature  

        '''
        dataset = self.train
        feature = self.get_dummy_variables()
        cur_features = pd.get_dummies(dataset[feature], drop_first=True)

        # drop one-hot feature if the 1s in the column is less than 10 
        to_drop = []
        for label, content in cur_features.iteritems():
            if (cur_features[label] == 1).sum() <= 10:
                to_drop.append(label)

        for i in range(len(to_drop)):
            cur_features = cur_features.drop(to_drop[i], 1)

        frames = [dataset, cur_features]
        new_dataset = pd.concat(frames, axis=1)
        return new_dataset.to_csv('preprocess_5.csv')

    def get_dummy_variables(self):
        return [
            'MSSubClass',
            'MSZoning',
            'Alley',
            'LandContour',
            'Utilities',
            'LotConfig',
            'Neighborhood',
            'Condition1',
            'Condition2',
            'BldgType',
            'HouseStyle',
            'RoofStyle',
            'RoofMatl',
            'Exterior1st',
            'Exterior2nd',
            'Foundation',
            'Heating',
            'Electrical',
            'GarageType',
            'Fence',
            'MiscFeature',
            'SaleType',
            'SaleCondition'
        ]

onehotEncoding = OneHotEncoding()
onehotEncoding.one_hot_encoded()


    

