''' This script uses GridSearchCV to test all possible combinations
of parameters for an ExtratreesClassifier and returns the one 
with the best parameters. '''

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from data_utils import load_train, load_test, write_test

import sys
sys.path.append("..")

if __name__ == '__main__':
	X, y = load_train('./data/train_2008.csv')

	param_grid = {
	    'n_estimators': [20, 50, 100, 500, 1000],
	    'criterion':  ['gini', 'entropy'],
	    'max_features': [None, 'auto', 'sqrt', 'log2'],
	    'max_depth': [2, 5, 10, 20]
	}

	etc = ExtraTreesClassifier()
	clf = GridSearchCV(estimator=etc, scoring='accuracy', param_grid=param_grid, 
	                   n_jobs=4, verbose=2)
	clf.fit(X, y)

	print('Best Error: %f' %(clf.best_score_))
	print('Best Model: %s' %(clf.best_params_))