import numpy
import keras
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

import sys, os
sys.path.append(os.path.abspath('..'))

from data_utils import load_train, load_test, write_test
from report import *

def create_model(x, y, neurons1=200, neurons2=50, neurons3 = 25, layer3 = 1, dropout=0.2, fa ='sigmoid', class_weight={0:1, 1:1}):
    # create model
    class_weight = {0 : 1.,
    1: 1.}
    model = Sequential()
    model.add(Dense(neurons1, input_dim=366))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(neurons2))
    model.add(Activation('relu'))
    if layer3 == 1:
        model.add(Dense(neurons3))
        model.add(Activation('relu'))
    model.add(Dense(2, activation = fa))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    model.fit(x, y, batch_size=500, nb_epoch=20, verbose = 0)
    return model
def paramSearch(x, y, xVal, yVal):
    dropout = [0.5]
    neurons1 = [100, 200, 150]
    neurons2 = [50, 100]
    neurons3 = [20]
    layer3 = [1, 0]
    classWeight = [{0:1, 1:1}]
    finalActivation = ['softmax', 'sigmoid']
    bestAccuracy = 0
    bestParams = []
    for d in dropout:
        for n1 in neurons1:
            for n2 in neurons2:
                for n3 in neurons3:
                    for l3 in layer3:
                        for f in finalActivation:
                            for c in classWeight:
                                model = create_model(x, y, neurons1=n1, neurons2=n2, neurons3 =n3, layer3 = l3, dropout=d, class_weight=c, fa = f)
                                score = model.evaluate(xVal, yVal)
                                print
                                print n1, n2, n3, l3, d, c, f
                                print('Test accuracy:', score[1])
                                if score[1] > bestAccuracy:
                                    bestParams = [d, n1, n2, n3, l3, f, c]
                                    bestAccuracy = score[1]
    print bestAccuracy
    print bestParams
def validatemodel(x, y):
    
    validationSize = 3
    xTrain = np.asarray(x[:2 * len(x) / 3])
    xValidation = np.asarray(x[2 * len(x) / 3:])
    yTrain = np.asarray(y[:2 * len(y) / 3])
    yValidation = np.asarray(y[2 * len(y) / 3:])
    
    paramSearch(xTrain, yTrain, xValidation, yValidation)
if __name__=='__main__':
    X, y = load_train('./data/train_2008.csv', True)
    X = np.asarray(X)
    y2 = []
    for val in y:
        y2.append(val-1)
    y2 = np.asarray(y2)
    np.asarray(y)
    yOneHot = keras.utils.np_utils.to_categorical(y2, nb_classes = 2)
    
    
    #max_weight = [0.5, 1, 2, 3, 4, 5]
    dropout = [0.0]
    neurons1 = [100, 200]
    neurons2 = [20, 50]
    neurons3 = [20]
    layer3 = [1, 0]
    validatemodel(X, yOneHot)
    
    