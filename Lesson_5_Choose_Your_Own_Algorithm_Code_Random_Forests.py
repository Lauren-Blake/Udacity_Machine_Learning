# Lesson 5: Choose Your Own Algorithm
# Section 6: Quiz: Choose-Your-Own Algorithm Checklist

#The goal of this script is to get a random forest algorithm up and running. 

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## RANDOM FOREST CLASSIFIER #################################


from sklearn.ensemble import RandomForestClassifier

# Define training set
X = features_train
Y = labels_train

clf = RandomForestClassifier(n_estimators=10)

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

# Note: The accuracy of this classifier is 0.91200000000000003. 
