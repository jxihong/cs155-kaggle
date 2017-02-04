Save your models here. You can use "pickle" or sci-kit learn's "joblib" to dump the bytes of your model to this folder.

*** Parameters are a result of Gridsearch for best cross-validation accuracy

*** Remember to turn warm_start=True when training ensemble methods so you can add more estimators later. I didn't know about that for the first few days.

1. randomforest100 - Random Forest Classifier with 100 trees, using entropy with max depth of 10 (0.778)

2. randforest100n - Same as above, on normalized data

3. linearsvm - SVM with linear kernel w C=0.1, and squared hinge loss (0.773)

4. rbfsvm - Bagged SVM with rbf kernel, using 20 estimators each trained on 10% of the data and features. Used C=0.1 and Gamma=0.15 (0.774)
 
5. randomforest1000 - Random Forest with 1000 trees, using entropy and max depth of 20 (0.781) 

6. neuralnet - Neural Network with 2 layers w/ 50 nodes in 1st hidden and 10 nodes in 2nd hidden, and 0.0 Dropoug (0.745)

7. gbtree - Gradient Boosted Tree with 200 estimators, and max depth of 5.

8. svm_kbest - SVM with rbf kernel, and feature classification to select the best 50 features. Used C=1.0 and gamma=0.165.

9. logistic - Basic logistic regression, using C=0.001.
