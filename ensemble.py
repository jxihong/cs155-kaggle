import numpy as np
from sklearn.metrics import accuracy_score

from multiprocessing import Pool
import os

class EnsembleClassifier(object):
    """
    Behaves as a voting classifier, and selects a model from a hypothesis set
    each iteration that maximizes prediction accuracy on a validation
    set.
    """
    
    def __init__(self, models=[], H=[], verbose=True):
        self.models = models
        self.n_models = 0

        self._H = H
        self.verbose = verbose
        

    def add_model(self, model):
        self.models.append(model)

    
    def get_models(self):
        return self.models
    

    def predict(self, x):
        classes = np.asarray([clf.predict(x) for clf in self.models])
        
        maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) 
                          for c in range(self.classes_.shape[1])])
        

    def log_loss(self, x, y):
        classes = np.asarray([clf.predict(x) for clf in self.models])
        
        # Calculate the log probabilities (want to maximize)
        (n, m) = classes.shape
        # Approximate probability of correct label as percent estimators 
        # that voted the correct label
        conf = np.sum(y == classes, axis=0)/ float(n)
        log_probs = -1 * np.log(conf.clip(min=1e-6))
        
        return np.mean(log_probs)    
    
    
    def acc(self, x, y):
        classes = np.asarray([clf.predict(x) for clf in self.models])
        
        maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) 
                          for c in range(self.classes_.shape[1])])
        
        return accuracy_score(y, maj)
    

    def fit_next_model(self, x_val, y_val, mode='hard'):
        best_h = None
        if mode == 'hard':
            best_s = 0.0
        else:
            best_s = 1e6

        np.random.shuffle(self._H) #shuffle the order of models
        for h in self._H:
            self.models.append(h)
            
            if mode == 'hard':
                s = self.acc(x_val, y_val)
                
                if (s > best_s):
                    best_s = s
                    best_h = models[i]
            else:
                s = self.log_loss(x_val, y_val)
            
                if (s < best_s):
                    best_s = s
                    best_h = h
                
            self.models.pop()

        if self.verbose:
            print 'Adding to Ensemble:%s' %(best_h)
            print 'Score: %f' %(best_s)
        
        self.models.append(best_h)
        self.n_models += 1
        
    

def train_models(models, X_train, y_train):
    """
    Models should contain a list of tuples, where the first element is a 
    unique id for the model (used as filename) and second is the model
    itself.
    Trains all the models on a set training set in parrallel.
    """

    def train_model(model):
        SAVE_DIR = 'models/ensemble_models'
        model[1].fit(X_train, y_train)
        
        print '%s - %s\n' %(model[0], model[1])
        joblib.dump(model[1], os.path.join(SAVE_DIR, '%s.pkl'%(model[0])))
        return
    
    # Pretty risky to run, my computer couldn't take it
    pool = Pool(4)
    pool.map(train_model, models)
    pool.close() 
    pool.join()
    
