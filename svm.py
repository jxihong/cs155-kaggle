import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from data_utils import load_train, load_test, write_test

if __name__ == '__main__':
    X, y = load_train('data/train_2008.csv')
    
    Cs = np.logspace(-7, 2, 10)
    gammas = np.logspace(-6, 2, 9, base=2)
    
    param_grid = [
        {'C': Cs,
         'kernel': ['linear']},
        {'C': Cs,
         'gamma': gammas,
         'kernel': ['rbf'],
         'cache_size': [5000] },
        {'C': Cs,
         'kernel': ['poly'],
         'degree' :[2]}
        ]
    
    clf = GridSearchCV(SVC(C=1), scoring='accuracy', param_grid=param_grid, 
                       n_jobs=2, verbose=2)
    
    clf.fit(X, y)

    
    print('Best Error: %f' %(clf.best_score_))

    joblib.dump(clf.best_estimator_, 'models/svm.pkl') 
    
    X_test = load_test('data/test_2008.csv')
    write_test('predictions/svm.csv', clf.best_estimator_.predict(X_test))

