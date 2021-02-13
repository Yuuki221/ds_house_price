import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel 
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score

import categorical_features
import get_preprocessed_data

class FeatureSelectionRandomForest:
    def __init__(self):
        self.CategoricalFeatures = categorical_features.CategoricalFeatures
        self.PreprocessTrainingData = get_preprocessed_data.PreprocessTrainData
        # get categorical features 
        self.preprocess_train_data = self.PreprocessTrainingData.get_preprocessed_data()
        self.categorical_features = self.CategoricalFeatures.get_categorical_features()

    def get_categorical_training_data(self):
        '''
           get categorical training dataset  
        '''
        return self.preprocess_train_data[self.categorical_features]

    def get_target_variable(self):
        '''
            get sale price from training dataset 
        '''
        return self.preprocess_train_data['SalePrice']

    def get_removed_categorical_features(self):
        '''
            based on random forest, get the categorical variables that is important 
        '''
        categorical_dataset = self.get_categorical_training_data()
        sale_price = self.get_target_variable()
        selection = SelectFromModel(RandomForestClassifier(n_estimators=100))
        selection.fit(categorical_dataset.fillna(0), sale_price)
        
        selection_feat = categorical_dataset.columns[(selection.get_support())]
        important_features = []
        for idx in range(len(selection_feat)):
            important_features.append(selection_feat[idx])
        return important_features

test = FeatureSelectionRandomForest()
print(test.get_removed_categorical_features())




    

    