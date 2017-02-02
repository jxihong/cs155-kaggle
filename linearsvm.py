import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from data_utils import load_train, load_test, write_test

if __name__ == '__main__':
    X, y = load_train('data/train_2008.csv')
    
    Cs = np.logspace(-7, 2, 10)
    
    param_grid = {'C': Cs,
                  'loss': ['hinge', 'squared_hinge'],
                  }
                  
    
    clf = GridSearchCV(LinearSVC(), scoring='accuracy',param_grid=param_grid, 
                       n_jobs=4, verbose=2)
    
    clf.fit(X, y)

    
    print('Best Error: %f' %(clf.best_score_))
    
    print 'Best Model:\n'
    print clf.best_estimator_

    joblib.dump(clf.best_estimator_, 'models/linearsvm.pkl') 
    
    X_test = load_test('data/test_2008.csv')
    write_test('predictions/linearsvm.csv', clf.best_estimator_.predict(X_test))

