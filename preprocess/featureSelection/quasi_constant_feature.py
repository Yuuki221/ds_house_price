# Quasi - Constant features 
import numpy as np

class Quasi:

    def get_quasi_constant_feature(preprocess_train):
        quasi_constant_feat = []
        for feature in preprocess_train:
            # find predominant value 
            predominant = (preprocess_train[feature].value_counts()/np.float(len(preprocess_train))).sort_values(ascending=False).values[0]
            # evaluate predominant
            if predominant > 0.999:
                quasi_constant_feat.append(feature)
                
        return quasi_constant_feat