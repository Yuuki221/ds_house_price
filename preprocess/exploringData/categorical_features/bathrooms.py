import pandas as pd 

class BathroomCounts:
    
    def create_newbathroom_var(self, dataset):
        dataset['TotalBathroom'] = dataset.apply(self.get_total_bathroom, axis=1)

    def get_total_bathroom(self, row):
        return row['FullBath'] + row['HalfBath']*0.5 + row['BsmtFullBath'] + row['BsmtHalfBath']*0.5

