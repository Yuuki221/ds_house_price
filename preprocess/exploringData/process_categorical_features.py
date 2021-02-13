import pandas as pd 
import seaborn as sns
import sys 
sys.path.insert(0, './categorical_features') 

from categorical_features import bathrooms
from categorical_features import consolidating_porch
from categorical_features import houseAge_remodel_isNew
from categorical_features import neighborhoods
from categorical_features import square_feet

class CategoricalFeatures:
    '''
        Deal with categorical features 
    '''
    def __init__(self):
        self.withoutMissingValues = pd.read_csv('~/ds_house/data/preprocess_2.csv', encoding='utf-8')
        self.Bathrooms = bathrooms.BathroomCounts()
        self.ConsolidatingPorch = consolidating_porch.ConsolidatingPorch()
        self.HouseAgeRemodelIsNew = houseAge_remodel_isNew.HouseAgeRemodelIsNew()
        self.Neighborhoods = neighborhoods.Neighborhoods()
        self.SquareFeet = square_feet.SquareFeet()


    def preprocess_categorical_features(self):
        '''
            preprocess categorical features 
        '''   

        # Categorical Feature Bathroom 
        # create new bathroom 
        self.Bathrooms.create_newbathroom_var(self.withoutMissingValues)
        # create consolidating porch
        self.ConsolidatingPorch.sum_porch_variables(self.withoutMissingValues)
        # create houseAgeRemodelIsNew 
        self.HouseAgeRemodelIsNew.create_remodel(self.withoutMissingValues)
        # create house age variable 
        self.HouseAgeRemodelIsNew.create_age(self.withoutMissingValues)
        # create isNew variable 
        self.HouseAgeRemodelIsNew.create_isNew(self.withoutMissingValues)
        # factorize neighborhoods 
        self.Neighborhoods.factor_neighbors(self.withoutMissingValues)
        # create total square feet 
        self.SquareFeet.create_totalSqFeet(self.withoutMissingValues)





