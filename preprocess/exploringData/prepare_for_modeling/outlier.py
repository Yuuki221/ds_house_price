import seaborn as sns 

class Outliers:
    def __init__(self):
        self.train = pd.read_csv('~/ds_house/data/train.csv', encoding='utf-8')
    
    def getRegressionPlot(self, x: str, y: str):
        '''
            Get regression scatter plot for variables that are highly related to SalePrice 
        '''
        sns.regplot(x=self.train['GrLivArea'], y=self.train['SalePrice'])
        sns.plot.show()

    def get_outliers(self, x_limit: int):
        '''
            Get outliers based on observation from scatter plot 
        '''
        outliers = set()
        
        for index, row in self.train.iterrows():
            if row['GrLivArea'] > 4500:
                outliers.add(index)
        return outliers

    
