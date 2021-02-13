import pandas as pd 

class PaveStreetDriveWay:
    '''
        Pave Street Drive Way Exploration 
    '''

    # def __init__(self):
    #     self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')

    def label_street(self, train):
        '''
            Re-label Street 
            'Street': {
                'NaN': 0,
                'Grvl': 0, 
                'Pave': 1
            }
        '''
        Street_Mapping = {
            'NaN': 0, 
            'Grvl': 0, 
            'Pave': 1
        }

        feature_df = train[['Street']]
        train['Street'] = feature_df.fillna('NaN').applymap(lambda x: Street_Mapping[x])


    def label_pavedDrive(self, train):
        '''
            Re-label PavedDrive
            'PavedDrive':{
                'NaN': 0, 
                'N': 0, 
                'P': 1,
                'Y': 2
            }
        '''

        PavedDrive_Mapping = {
            'NaN': 0, 
            'N': 0, 
            'P': 1, 
            'Y': 2
        }

        feature_df = train[['PavedDrive']]
        train['PavedDrive'] = feature_df.fillna('NaN').applymap(lambda x: PavedDrive_Mapping[x])