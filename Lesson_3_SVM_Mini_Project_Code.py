#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
# Import SVM

from sklearn.svm import SVC

# Import numpy

import numpy as np

# Define X and Y (first two lines means only use the 1% of the training data)

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

X = features_train
y = labels_train

# Define a linear SVM
# We can specify a linear or a RBF kernel with a given C parameter
clf = SVC(kernel="rbf", C = 10000)

#  fit the classifier
t0 = time()
clf.fit(X, y)
print "training time:", round(time()-t0, 3), "s"

#### store your predictions in a list named pred
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

# Extracting Predictions From An SVM

answer1=pred[10]
answer2=pred[26]
answer3=pred[50]

#print(answer1)
#print(answer2)
#print(answer3)

answer=np.array([answer1, answer2, answer3])
print(answer)

# How many emails is Chris predicted to have written?

chris = np.count_nonzero(pred)
print(chris)

# Measure accuracy of the classifier

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, labels_test)

print(acc)


