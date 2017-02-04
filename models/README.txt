Save your models here. You can use "pickle" or sci-kit learn's "joblib" to dump the bytes of your model to this folder.

*** Parameters are a result of Gridsearch for best cross-validation accuracy

1. randomforest100.pkl - Random Forest Classifier with 100 trees, using entropy with max depth of 10 (0.778)

2. randforest100n.pkl - Same as above, on normalized data

3. linearsvm.pkl - SVM with linear kernel w C=0.1, and squared hinge loss (0.773)

4. rbfsvm.pkl - Bagged SVM with rbf kernel, using 20 estimators each trained on 10% of the data and features. Used C=0.1 and Gamma=0.15 (0.774)
 
5. randomforest1000.pkl - Random Forest with 1000 trees, using entropy and max depth of 20 (0.781) 

6. neuralnet.pkl - Neural Network with 2 layers w/ 50 nodes in 1st hidden and 10 nodes in 2nd hidden, and 0.0 Dropoug (0.745)
