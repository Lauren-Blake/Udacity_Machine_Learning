# Lesson 4: Decision Trees
# Section 12: Decision Tree Accuracy

#The goal of this script is to obtain the accuracy of the decision boundary for the training and test data provided. There are two classifiers: min_sample_split = 2 and  min_sample_split = 2.




import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



########################## DECISION TREE #################################


### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively

# Insert studentMain.py from Lesson 4 Section 9 Decision Tree Accuracy

from sklearn import tree

# Define training set
X = features_train
Y = labels_train

###### Select min_samples_split = 2 #######

clf = tree.DecisionTreeClassifier(min_samples_split=2)

#Fit the classifier

clf = clf.fit(X,Y)

#Make predictions using the classifer and training data
pred = clf.predict(features_test)

#clf = classify(features_train, labels_train)


# Get the accuracy from previous mini-projects
from sklearn.metrics import accuracy_score

acc_min_samples_split_2 = accuracy_score(pred, labels_test)
print(acc_min_samples_split_2)

###### Select min_samples_split = 50 #######

clf = tree.DecisionTreeClassifier(min_samples_split=50)

#Fit the classifier

clf = clf.fit(X,Y)

#Make predictions using the classifer and training data
pred = clf.predict(features_test)

#clf = classify(features_train, labels_train)


# Get the accuracy from previous mini-projects
from sklearn.metrics import accuracy_score

acc_min_samples_split_50 = accuracy_score(pred, labels_test)
print(acc_min_samples_split_50)

