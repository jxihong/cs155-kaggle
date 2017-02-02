import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from data_utils import load_train, load_test, write_test

if __name__ == '__main__':
    X, y = load_train('data/train_2008.csv')
    
    param_grid = {
        'n_estimators': [20, 50, 100],
        'criterion':  ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [1, 2, 5, 10]
    }

    rfc = ens.RandomForestClassifier()
    clf = GridSearchCV(estimator=rfc, scoring='accuracy', param_grid=param_grid, n_jobs=4, verbose=2)
    clf.fit(X, y)

    print('Best Error: %f' %(clf.best_score_))

    joblib.dump(clf.best_estimator_, 'models/randomforest.pkl') 
    
    X_test = load_test('data/test_2008.csv')
    write_test('predictions/randomforest.csv', clf.best_estimator_.predict(X_test))
    
    
