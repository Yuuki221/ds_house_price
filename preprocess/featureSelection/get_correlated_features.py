# Get correlated feature 
class CorrelationFeatures:
    def __init__(self):
        self.grouped_feature_ls = []
        self.correlated_groups = []

    def correlation(self, dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: 
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr
    
    def get_correlated_features(self, correlation_data, corr_rate):
        return self.correlation(self, correlation_data, corr_rate)
