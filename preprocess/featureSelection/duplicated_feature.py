# Duplicated Features 
class DuplicatedFeatures:

    def get_duplicate_features(preprocess_train):
        # Duplicated Features 
        duplicated_feat = []
        for i in range(0, len(preprocess_train.columns)-1):
            col_1 = preprocess_train.columns[i]
            for col_2 in preprocess_train.columns[i+1:]:
                # if the feature are duplicate 
                if preprocess_train[col_1].equals(preprocess_train[col_2]):
                    duplicated_feat.append(col_2)
        return duplicated_feat