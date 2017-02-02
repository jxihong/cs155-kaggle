import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

from data_utils import load_train, load_test, write_test

if __name__ == '__main__':
    X, y = load_train('data/train_2008.csv')
    
    Cs = np.logspace(-7, 2, 10)
    gammas = np.logspace(-6, 2, 9, base=2)

    best_score = 0
    best_model = None
    for p in zip(Cs, gammas):
        bagging = BaggingClassifier(SVC(C=p[0],cache_size=7000, kernel='rbf', gamma=p[1]), 
                                    n_estimators=20, max_samples=0.1, max_features=0.1)
        scores = cross_val_score(bagging, X, y, scoring='accuracy', cv=3)        
        avg_score = scores.mean()
        
        print ('C=%s, gamma=%s, cross_val_score=%s' %(p[0], p[1], avg_score))
        
        if best_score <= avg_score:
            best_score = avg_score
            best_model = bagging

    
    print('Best Error: %f' %(best_score))

    joblib.dump(best_model, 'models/rbfsvm.pkl') 
    
    X_test = load_test('data/test_2008.csv')
    write_test('predictions/rbfsvm.csv', best_model.predict(X_test))

