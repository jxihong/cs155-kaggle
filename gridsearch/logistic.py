import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

import sys, os
sys.path.append(os.path.abspath('..'))

from data_utils import load_train, load_test, write_test
from report import *

if __name__ == '__main__':
    X, y = load_train('../data/train_2008.csv')
    
    clf = LogisticRegression()    
    
    Cs = np.logspace(-4, 4, 5)
    
    search = GridSearchCV(clf, dict(C=Cs))
    search.fit(X, y)
    
    report(search.cv_results_)
    
    print('Best Error: %f' %(search.best_score_))
    print('Best Model: %s' %(search.best_params_))
