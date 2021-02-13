# Constant Features 

'''
    remove the feature colimns which have same value in all the data 
    where variance is zero 
'''
class ConstantFeature:
    def constant_feature(preprocess_train, numerical_features):
        constant_features = [features for features in numerical_features if preprocess_train[features].std() == 0]
        return constant_features 