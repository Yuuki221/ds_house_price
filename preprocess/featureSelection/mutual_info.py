from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile 
import pandas as pd
import math

class MutualInfo:
    def get_mutual_info(preprocess_train, numerical_features, delete_percentage):
        sale_price = preprocess_train['SalePrice']
        correlation_data = preprocess_train[numerical_features]
        mutual_info = mutual_info_regression(correlation_data.fillna(0), sale_price)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = correlation_data.columns
        mutual_info = mutual_info.sort_values(ascending=True)
        delete_num = math.floor(len(mutual_info)*delete_percentage)
        return mutual_info.index[:delete_num]

    def get_MI_plot(mutual_info):
        mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))