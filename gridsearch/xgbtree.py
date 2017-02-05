import numpy as np

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import sys, os
sys.path.append(os.path.abspath('..'))

from data_utils import load_train, load_test, write_test
from report import *

if __name__=='__main__':
    X, y = load_train('../data/train_2008.csv', False)
    
    fix_params = {'learning_rate': 0.1, 
                  'n_estimators': 1000, 
                  'objective': 'binary:logistic'
                  }
    
    param_grid = {'max_depth': [3,5,7,10], 
                  'colsample_bytree': [0.5, 0.8, 0.9],
                  'subsample': [0.5, 0.8, 0.9]
                  }

    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**fix_params), param_grid=param_grid, 
                                 scoring='accuracy', 
                                 n_jobs=4, verbose=2) 
    
    optimized_GBM.fit(X, y)
    
    report(optimized_GBM.cv_results_)

    print('Best Error: %f' %(optimized_GBM.best_score_))
    print('Best Model: %s' %(optimized_GBM.best_params_))
