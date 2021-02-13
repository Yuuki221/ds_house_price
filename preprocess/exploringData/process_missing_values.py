import pandas as pd 
import sys
import seaborn as sns
import sys
sys.path.insert(0, './missingValues') 
from missingValues import pool_quality
from missingValues import misc_feature
from missingValues import alley
from missingValues import fence 
from missingValues import fireplace
from missingValues import lot_features
from missingValues import garage
from missingValues import basement_features
from missingValues import masory_features
from missingValues import mszoning
from missingValues import kitchen_features
from missingValues import utilities
from missingValues import functional_features
from missingValues import exterior_features
from missingValues import electrical

class MissingValue:
    '''
        Deal with missing values 
    '''
    def __init__(self):
        self.train = pd.read_csv('~/ds_house/preprocess_7.csv', encoding='utf-8')
        self.PoolQuality = pool_quality.PoolQuality()
        self.MiscFeature = misc_feature.MiscFeature()
        self.Alley = alley.Alley()
        self.Fence = fence.Fence()
        self.Fireplace = fireplace.Fireplace()
        self.LotFeatures = lot_features.LotFeatures()
        self.Garage = garage.GarageData()
        self.BasementFeatures = basement_features.BasementFeatures()
        self.MasoryFeatures = masory_features.MasoryFeatures()
        self.MSZoning = mszoning.MSZoning()
        self.KitchenFeatures = kitchen_features.KitchenFeatures()
        self.Utilities = utilities.Utilities()
        self.Functional = functional_features.Functional()
        self.ExteriorFeatures = exterior_features.ExteriorFeatures()
        self.Electrical = electrical.Electrical()


    def preprocess_dataset(self):
        '''
            Preprocess dataset 
        '''
        # Pool quality 
        # self.PoolQuality.label_pool_quality(self.train)
        # Miscellaneous Features
        # alley
        # fire place
        # self.Fireplace.label_fireplace_quality(self.train)
        # lot features 
        # self.LotFeatures.replace_lotFrontage_with_median(self.train)

        # Garage 
        self.Garage.impute_GarageYrBlt_with_YearBuilt(self.train)
        self.Garage.impute_other_garage_with_mode(self.train)
        self.Garage.relabel_garageFinish(self.train)
        self.Garage.relabel_garageQual(self.train)
        self.Garage.relabel_garageCond(self.train)

        # basement features 
        self.BasementFeatures.relabel_bsmtqual(self.train)
        self.BasementFeatures.relabel_bsmtcond(self.train)
        self.BasementFeatures.relabel_bsmtExposure(self.train)
        self.BasementFeatures.relabel_bsmtFinType1(self.train)
        self.BasementFeatures.relabel_bsmtFinType2(self.train)

        # Masory features 
        self.MasoryFeatures.label_masVnrType(self.train)
        self.MasoryFeatures.impute_masVnrArea(self.train)

        # MSZoning features 
        # Kitchen features 
        self.KitchenFeatures.label_kitchen_quality(self.train)

        # remove utilities 
        self.Utilities.remove_utilities(self.train)

        # functional features 
        self.Functional.relabel_functional(self.train)

        # exterior feature 
        self.ExteriorFeatures.relabel_exterQual(self.train)
        self.ExteriorFeatures.relabel_exterCond(self.train)

        # electrical features 
        self.Electrical.impute_eletrical_value(self.train)

missing_values = MissingValue()
missing_values.preprocess_dataset()

         
        

    