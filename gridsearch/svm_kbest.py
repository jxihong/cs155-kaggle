import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

import sys, os
sys.path.append(os.path.abspath('..'))

from data_utils import load_train, load_test, write_test
from report import *

# Can't use GridSearchCV because the SVC parameters are nested inside Bagging Classifier
if __name__ == '__main__':
    X, y = load_train('../data/train_2008.csv')
    
    Cs = np.logspace(-7, 2, 10)
    gammas = np.logspace(-6, 2, 9, base=2)

    filter = SelectKBest(f_classif, k=50)
    clf = SVC(kernel='rbf')
    
    pipe = Pipeline([('filter', filter), ('svc', clf)])
    
    search = GridSearchCV(pipe,
                          dict(svc__C=Cs, svc__gamma = gammas), n_jobs=4, verbose=2)
    search.fit(X, y)
        
    report(search.cv_results_)
    
    print('Best Error: %f' %(search.best_score_))
    print('Best Model: %s' %(search.best_params_))
    
