# Lesson 4: Decision Trees
# Section 9: Decision Tree Accuracy

The goal of this script is to obtain the accuracy of the decision boundary for the training and test data provided.


import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################

# Insert studentMain.py from Lesson 4 Section 8 Coding a Decision Tree

from sklearn import tree

# Define training set
X = features_train
Y = labels_train

clf = tree.DecisionTreeClassifier()

#Fit the classifier

clf = clf.fit(X,Y)

#Make predictions using the classifer and training data
pred = clf.predict(features_test)

#clf = classify(features_train, labels_train)


# Get the accuracy from previous mini-projects
from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, labels_test)

#print(acc)
    
def submitAccuracies():
  return {"acc":round(acc,3)}
