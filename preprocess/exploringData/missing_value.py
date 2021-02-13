class MissingValues:
    def __init__(self):
        self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')
    
    def get_missing_value(self):
        '''
            get missing value set 
        '''
        missing_vals = dict()
        for index, val in train.isnull().sum(axis=0).iteritems():
            if val > 0:
                missing_vals[index] = val

        return missing_vals