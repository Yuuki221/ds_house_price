import numpy as np 
import pandas as pd 
import csv 

class PreprocessTrainingSet:
    # import row training data and testing data 
    def __init__(self):
        self.train = pd.read_csv('~/ds_house/preprocess_3.csv', encoding='utf-8')
        #self.test = pd.read_csv('~/ds_house/data/test.csv', encoding='utf-8')
    
    def ordinal_encoding(self):
        '''
            apply ordinal encoding to fields of training set 
            
            HeatingQC 
            CentralAir
            LandSlope
            Street
            PavedDrive
            PoolQC 
            FireplaceQu
            LotShape
            GarageFinish
            GarageQual
            GarageCond
            BsmtQual
            BsmtCond
            BsmtExposure
            BsmtFinType1
            BsmtFinType2
            MasVnrType
            KitchenQual
            Functional
            ExterQual
            ExterCond

        '''
        ordinal_encoding_features = [
            'HeatingQC', 
            'CentralAir',
            'LandSlope',
            'Street',
            'PavedDrive',
            #'PoolQC',
            #'FireplaceQu',
            'LotShape',
            #'GarageFinish',
            #'GarageQual',
            #'GarageCond',
            #'BsmtQual',
            #'BsmtCond',
            #'BsmtExposure',
            #'BsmtFinType1',
            #'BsmtFinType2',
            #'MasVnrType',
            #'KitchenQual',
            #'Functional',
            #'ExterQual',
            #'ExterCond'
        ]

        encoding_mapping = {    
            'HeatingQC': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'CentralAir': {
                'N': 0,
                'Y': 1
            },
            'LandSlope': {
                'Sev': 0,
                'Mod': 1,
                'Gtl': 2
            },
            'Street': {
                'Grvl': 0,
                'Pave': 1
            },
            'PavedDrive': {
                'N': 0,
                'P': 1,
                'Y': 2 
            },
             'PoolQC': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'FireplaceQu': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'LotShape': {
                'IR3': 0, 
                'IR2': 1,
                'IR1': 2,
                'Reg': 3
            },
            'GarageFinish': {
                'Fin': 3, 
                'RFn': 2,
                'Unf': 1,
                'NaN': 0
            },
            'GarageQual': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'GarageCond': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'BsmtQual': {
                'NaN': 0,
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'BsmtCond': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'BsmtExposure': {
                'NaN': 0,
                'No': 1, 
                'Mn': 2,
                'Av': 3,
                'Gd': 4
            },
            'BsmtFinType1': {
                'NaN': 0, 
                'Unf': 1,
                'LwQ': 2,
                'Rec': 3,
                'BLQ': 4, 
                'ALQ': 5,
                'GLQ': 6
            },
            'BsmtFinType2': {
                'NaN': 0, 
                'Unf': 1,
                'LwQ': 2,
                'Rec': 3,
                'BLQ': 4, 
                'ALQ': 5,
                'GLQ': 6
            },
            'MasVnrType': {
                'NA': 0,
                'BrkCmn': 0,
                'BrkFace': 1,
                'Stone': 2
            },
            'KitchenQual': {
                'NaN': 0, 
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'Functional': {
                'Sal': 0,
                'Sev': 1,
                'Maj2': 2,
                'Maj1': 3,
                'Mod': 4,
                'Min2': 5,
                'Min1': 6,
                'Typ': 7 
            },        
            'ExterQual': {
                'Po': 1,
                'Fa': 2,
                'TA': 3,
                'Gd': 4,
                'Ex': 5
            },
            'ExterCond': {
                'Po': 1,
                'Fa': 2, 
                'TA': 3,
                'Gd': 4,
                'Ex': 5 
            }
        }

        for i in range(0, len(ordinal_encoding_features)):
            feature = ordinal_encoding_features[i]    
            self.apply_categorical_mapping(feature, encoding_mapping[feature])


    def apply_categorical_mapping(self, feature: str, mapping: object):
        '''
            apply categorical mapping to different categories 
        '''
        feature_df = self.train[[feature]]
        self.train[feature] = feature_df.fillna('NaN').applymap(lambda x : mapping[x])


    def revalue_mssubclass(self):
        MSSubclass_Mapping = {
            20: '1 story 1946+',
            30: '1 story 1945-',
            40: '1 story unf attic',
            45: '1,5 story unf',
            50: '1,5 story fin',
            60: '2 story 1946+',
            70: '2 story 1945-',
            75: '2,5 story all ages',
            80: 'split/multi level',
            85: 'split foyer',
            90: 'duplex all style/age',
            120: '1 story PUD 1946+',
            150: '1,5 story PUD all',
            160: '2 story PUD 1946+',
            180: 'PUD multilevel',
            190: '2 family conversion'
        }
        feature_df = self.train[['MSSubClass']]
        values = feature_df.applymap(lambda x : MSSubclass_Mapping[x])
        self.train[['MSSubClass']] = values 

    def export_preprocess_data(self):
        self.revalue_mssubclass()
        self.ordinal_encoding()
        return self.train.to_csv('preprocess_4.csv')

PreprocessTraining = PreprocessTrainingSet()
PreprocessTraining.export_preprocess_data()



    




