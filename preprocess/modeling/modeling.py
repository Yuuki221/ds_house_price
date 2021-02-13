import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import joblib

class Modeling:
    '''
        different modeling on the preprocessed data 
    '''
    def __init__(self):
        self.training = pd.read_csv('~/ds_house/preprocess_7.csv', encoding='utf-8')

    def rmse(self, y_true, y_pred):
        '''
            Compute RMSE value based on true y value and predicted y value 
        '''
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def apply_lasso_modeling(self):
        '''
            Apply Lasso Model on training set 
        '''
        x_dataset = self.training 
        x_dataset = self.training.loc[:, ~self.training.columns.str.contains('^Unnamed')]
        del x_dataset['Id']
        print(len(x_dataset.columns.tolist()))
        label_df = pd.DataFrame(index = x_dataset.index, columns=['SalePrice'])
    
        label_df = pd.DataFrame(x_dataset[['SalePrice']])
        best_alpha = 0.00099
        regr = Lasso(alpha=best_alpha, max_iter=50000)
        regr.fit(x_dataset, label_df)

        y_pred = regr.predict(x_dataset)
        y_test = label_df 
        rmse_vals = self.rmse(y_test, y_pred)
        y_pred_lasso = regr.predict(x_dataset)

        # save model to joblib file 
        filename = 'lasso_model.joblib'
        joblib.dump(regr, filename)

        return {
            'rmse': rmse_vals,
            'y_pred': y_pred_lasso
        }

    def XGBoost_modeling(self):
        '''
            use XGBoost model to apply to the dataset 

        '''
        x_dataset = self.training

        ## parameters 
        params = {
            'colsample_bytree': 0.2,
            'gamma': 0.0,
            'learning_rate': 0.01,
            'max_depth': 4,
            'min_child_weight': 1.5,
            'n_estimator': 7200,
            'reg_alpha': 0.9,
            'reg_lambda': 0.6,
            'subsamples': 0.2,
            'seed': 42,
            'silent': 1
        }

        label_df = pd.DataFrame(x_dataset[['SalePrice']])
        del x_dataset['SalePrice']
        matrix = xgb.DMatrix(x_dataset, label_df)

        regr = xgb.cv(
            params,
            matrix,
            num_boost_round=500,
            nfold=3,
            stratified=False,
            folds=None,
            metrics=(),
            obj=None,
            feval=None,
            maximize=None,
            early_stopping_rounds=10,
            fpreproc=None,
            as_pandas=True,
            verbose_eval=None,
            show_stdv=True,
            seed=0,
            callbacks=None,
            shuffle=True
        )
        print(regr)

modeling = Modeling()
#modeling.apply_lasso_modeling()
modeling.XGBoost_modeling()


