# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:05:20 2017

@author: Badrinath
"""

import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

train_url = "train.csv"
train = pd.read_csv(train_url)

test_url = "test.csv"
test = pd.read_csv(test_url)

#clean test and train
train["Sex_float"] = float('NaN')
train["Sex_float"][train["Sex"] == "female"] = 1
train["Sex_float"][train["Sex"] == "male"] = 0
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Age"] = train["Age"].fillna(train["Age"].median())    
train["family_size"] = train["Parch"] + train["SibSp"] + 1   
     
test["Sex_float"] = float('NaN')
test["Sex_float"][test["Sex"] == "female"] = 1
test["Sex_float"][test["Sex"] == "male"] = 0
test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Age"] = test["Age"].fillna(test["Age"].median())
test["family_size"] = test["Parch"] + test["SibSp"] + 1
test.Fare[152] = test.Fare.dropna().median() 



# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex_float", "Fare", "SibSp", "Parch", "Embarked","family_size"]].values
# Building and fitting my_forest
forest = RandomForestClassifier(max_features =4, n_estimators = 300, max_depth = 8, min_samples_leaf = 3)
my_forest = forest.fit(features_forest,train["Survived"].values)

#test score with train
print(my_forest.feature_importances_)
print("train prediction score random forest:", my_forest.score(features_forest, train["Survived"].values))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex_float", "Fare", "SibSp", "Parch", "Embarked" ,"family_size"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_forest = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
my_solution_forest.to_csv("my_solution_forest.csv", index_label = ["PassengerId"])
#print(my_solution)
# Check that your data frame has 418 entries
print(my_solution_forest.shape)


features_xgb = train[["Pclass", "Age", "Sex_float", "Fare", "SibSp", "Parch", "Embarked","family_size"]].values
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=.04)
gbm = gbm.fit(features_xgb,train["Survived"].values)
print("train prediction score xgb:", gbm.score(features_xgb, train["Survived"].values))

predictions_xgb = gbm.predict(test_features)
my_solution_xgb = pd.DataFrame(predictions_xgb, PassengerId, columns = ["Survived"])
my_solution_xgb.to_csv("my_solution_xgb.csv", index_label = ["PassengerId"])
