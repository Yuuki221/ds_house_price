# get correlation matrix 
class CorrelationMatrix:
    def get_correlation_matrix(preprocess_train, numerical_features):
        correlation_data = preprocess_train[numerical_features]
        corrmat = correlation_data.corr()
        return corrmat