import numpy as np

from scipy.stats import randint as sp_rand

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

import sys, os
sys.path.append(os.path.abspath('..'))

from data_utils import load_train, load_test, write_test
from report import *

if __name__=='__main__':
    X, y = load_train('../data/train_2008.csv', False)

    gbm = GradientBoostingClassifier()
    
    param_grid = {
        'learning_rate': [0.01, 0.5, 0.1],
        'max_depth': [2, 5, 10],
        'subsample': [0.8, 0.9],
        'max_features': [20, 50, 100],
    }
    
    clf = GridSearchCV(gbm, scoring='accuracy', param_grid=param_grid, 
                       n_jobs=4, verbose=2)
    clf.fit(X, y)
    
    report(clf.cv_results_)
   
    print('Best Error: %f' %(clf.best_score_))
    print('Best Model: %s' %(clf.best_params_))
