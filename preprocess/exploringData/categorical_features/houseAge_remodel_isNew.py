import pandas as pd 

class HouseAgeRemodelIsNew:
    '''
        Create a remodel variable, which indicate the if the 
        house has been remodeled or not 
    '''
    def create_remodel(self, dataset):
        if 'Remodel' not in dataset:
            dataset['Remodel'] = dataset.apply(self.get_remodel_variable, axis=1)

    def get_remodel_variable(self, row):
        if row['YearBuilt'] == row['YearRemodAdd']:
            return 0 
        else:
            return 1


    '''
        Create age variable, indicating the house's age 
    '''
    def create_age(self, dataset):
        if 'Age' not in dataset:
            dataset['Age'] = dataset.apply(self.get_age_variable, axis=1)

    def get_age_variable(self, row):
        return row['YrSold'] - row['YearRemodAdd']


    '''
        Create isNew indicating the house is newly built or not 
    '''
    def create_isNew(self, dataset):
        if 'IsNew' not in dataset:
            dataset['IsNew'] = dataset.apply(self.get_isNew_variable, axis=1)

    def get_isNew_variable(self, row):
        if row['YrSold'] == row['YearBuilt']:
            return 1 
        else:
            return 0 

    

    