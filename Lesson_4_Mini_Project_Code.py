#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
# Import additional libraries

import numpy as np
import pylab as pl
from sklearn import tree

# Define training set
X = features_train
Y = labels_train

# Ask about the number of features in our data

print(len(features_train[0]))

# Get and then fit the classifier with given min_samples_split (with the time for training)
t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(X,Y)
print "training time:", round(time()-t0, 3), "s"

#Make predictions using the classifer and test data (with printing the time for making predictions)
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

# Get the accuracy from previous mini-projects
from sklearn.metrics import accuracy_score

acc_min_samples_split_40 = accuracy_score(pred, labels_test)
print(acc_min_samples_split_40)






