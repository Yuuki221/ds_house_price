# Get the categorical features for training data 
class CategoricalFeatures:
    def get_categorical_features():
        return [
            'BsmtQual_Mapping','BsmtCond_Mapping','BsmtExposure_Mapping','BsmtFinType1_Mapping',
            'BsmtFinType2_Mapping','ExterQual_Mapping','ExterCond_Mapping','Electrical_Mapping',
            'FireplaceQu_Mapping','Fence_Mapping','GarageQual_Mapping','GarageCond_Mapping',
            'GarageFinish_Mapping','HeatingQC_Mapping','KitchenQual_Mapping',
            'LotShape_Mapping','PoolQC_Mapping','Utilities_Mapping',
            'MSSubClass_Recategory', 'OverallQual_Recategory', 'OverallCond_Recategory'
        ]
