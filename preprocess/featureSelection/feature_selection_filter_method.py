import get_preprocessed_data
import quasi_constant_feature
import duplicated_feature
import numerical_features
import get_correlation_matrix
import get_correlated_features
import mutual_info
import constant_feature

class FeatureSelectionFilter:
    def __init__(self):
        self.PreprocessTrainData = get_preprocessed_data.PreprocessTrainData
        self.Quasi = quasi_constant_feature.Quasi
        self.DuplicatedFeatures = duplicated_feature.DuplicatedFeatures
        self.NumerocalFeatures = numerical_features.NumericalFeatures
        self.CorrelationMatrix = get_correlation_matrix.CorrelationMatrix
        self.CorrelationFeatures = get_correlated_features.CorrelationFeatures
        self.MutualInfo = mutual_info.MutualInfo
        self.ConstantFeatures = constant_feature.ConstantFeature
        # get preprocessed train data from its module 
        self.preprocess_train_data = self.PreprocessTrainData.get_preprocessed_data()
        self.numerical_features = self.NumerocalFeatures.get_numerical_features()
        
    # handle constant features 
    def check_constant_features(self):
        '''
            process constant features 
            if constant feature exists return them else return False 
        '''
        training_data = self.preprocess_train_data
        self.constant_features = self.ConstantFeatures.constant_feature(training_data, self.numerical_features)
        if len(self.constant_features) == 0:
            # no constant features 
            return False
        else:
            # return constant features 
            return self.constant_features

    def check_quasi_features(self):
        '''
            process features for quasi filter 
        '''
        self.quasi_features = self.Quasi.get_quasi_constant_feature(self.preprocess_train_data)
        if len(self.quasi_features) == 0:
            return False
        else:
            return self.quasi_features

    def check_duplicated_features(self):
        '''
            check if there are duplicate variables 
        '''
        self.duplicated_features = self.DuplicatedFeatures.get_duplicate_features(self.preprocess_train_data)
        if len(self.duplicated_features) == 0:
            return False
        else:
            return self.duplicated_features

    def get_heatmap(self):
        '''
            Get heat map 
        '''
        corr_matrix = self.CorrelationMatrix.get_correlation_matrix(self.preprocess_train_data, self.numerical_features)
        # generate heapmap 
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 16)
        sns.heatmap(corr_matrix)
    
    def get_dropped_correlated_vars(self):
        '''
            Drop correlated variables 
        '''
        correlated_features = self.CorrelationFeatures.get_correlated_features(self.CorrelationFeatures, self.preprocess_train_data[self.numerical_features], 0.8)
        
        return correlated_features

    def check_mutual_info_percentile(self, percentile=10):
        # corr_matrix = self.CorrelationMatrix.get_correlation_matrix(self.preprocess_train_data, self.numerical_features)
        # print(corr_matrix)
        mutual_info = self.MutualInfo.get_mutual_info(self.preprocess_train_data, self.numerical_features, percentile/100)
        return mutual_info

    def filter_methods_start(self):
        features_to_remove = []
        # explore constant features 
        constant_features = self.check_constant_features()
        if constant_features:
            for feature in constant_features:
                features_to_remove.append(feature)

        # explore quasi 
        quasi_features = self.check_quasi_features()
        if quasi_features:
            for feature in quasi_features:
                features_to_remove.append(feature)

        # get correlated features 
        correlated_features = self.get_dropped_correlated_vars()
        if correlated_features:
            for feature in correlated_features:
                features_to_remove.append(feature)
        
        # handle mutual information 
        mutual_info_features = self.check_mutual_info_percentile(20)
        for idx in range(len(mutual_info_features)):
            features_to_remove.append(mutual_info_features[idx])
        
        # remove unnecessary categorical column
        # final_dataset = self.preprocess_train_data.drop(columns=features_to_remove)
        return features_to_remove

test = FeatureSelectionFilter()
print(test.filter_methods_start())

            
        

        
        
            


    
        




        



    





        

    


    
        

    
