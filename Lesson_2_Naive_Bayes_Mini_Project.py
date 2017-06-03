#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

 ### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB

### create classifier
clf = GaussianNB()

### train classifier (with timing)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

    ### use the trained classifier to predict labels for the test features
t0 = time()
clf_pf = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

### import an sklearn module
from sklearn.metrics import accuracy_score
    
accuracy = accuracy_score(labels_test, clf_pf)
print(accuracy)



