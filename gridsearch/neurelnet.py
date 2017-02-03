import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier

from data_utils import load_train, load_test, write_test

def create_model(neurons=10, dropout=0.0, max_weight=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=381, init='normal',
                    activation='relu', W_constraint=maxnorm(max_weight)))
    model.add(Dropout(dropout))
    model.add(Dense(10, init='normal', activation='relu',
                    W_constraint=maxnorm(max_weight)))
    model.add(Dropout(dropout))
    model.add(Dense(1, init='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    return model


if __name__=='__main__':
    X, y = load_train('../data/train_2008.csv', False)
    
    model = KerasClassifier(build_fn=create_model, nb_epoch=20, 
                            batch_size=10, verbose=0)
    
    max_weight = [0.5, 1, 2, 3, 4, 5]
    dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    neurons = [10, 20, 50, 100, 200, 500]
    param_grid = dict(neurons=neurons, dropout=dropout, 
                      max_weight=max_weight) 

    clf = GridSearchCV(estimator=model, scoring='accuracy', param_grid=param_grid, 
                       n_jobs=4, verbose=2)
    clf.fit(X, y)
    
    print('Best Error: %f' %(clf.best_score_))
    print('Best Model: %s' %(clf.best_params_))
    
