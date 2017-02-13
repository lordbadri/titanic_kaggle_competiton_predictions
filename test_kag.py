#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 23:30:02 2017

@author: badrinath
"""
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    

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
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)  
train["family_size"] = train["Parch"] + train["SibSp"] + 1   
     
test["Sex_float"] = float('NaN')
test["Sex_float"][test["Sex"] == "female"] = 1
test["Sex_float"][test["Sex"] == "male"] = 0
test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test["family_size"] = test["Parch"] + test["SibSp"] + 1
test.Fare[152] = test.Fare.dropna().median() 

X = train[["Pclass", "Age", "Sex_float", "Fare", "SibSp", "Parch", "Embarked","family_size"]]
y = train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

# Building and fitting my_forest
#1forest = RandomForestClassifier(max_features = 5, n_estimators = 10000, max_depth = 20, min_samples_leaf = 3)
forest = RandomForestClassifier(max_features = 4, n_estimators = 10000, max_depth = 15, min_samples_leaf = 10)
my_forest = forest.fit(X_train, y_train)
predicted = my_forest.predict(X_val)
#test score with train
#print(my_forest.feature_importances_)
print metrics.accuracy_score(y_val, predicted)

#test values Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex_float", "Fare", "SibSp", "Parch", "Embarked" ,"family_size"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_forest_test= pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
my_solution_forest_test.to_csv("my_solution_forest_test.csv", index_label = ["PassengerId"])


# Example code for a model and a set of grid-search parameters
model = RandomForestClassifier()
parameters = {"n_estimators"      : [200, 500, 1000],
           "max_features"      : [3, 4, 5],
           "max_depth"         : [10, 15, 20],
           "min_samples_leaf" : [2, 3, 4]}

 
# Returns the best configuration for a model using crosvalidation
# and grid search

def best_config(model, parameters, train_instances, judgements):
    clf = GridSearchCV(model, parameters, cv=5,
                       scoring="accuracy", verbose=5, n_jobs=4)
    clf.fit(train_instances, judgements)
    best_estimator = clf.best_estimator_
    return [str(clf.best_params_), clf.best_score_,best_estimator]
    
# Returns the best model from a set of model families given
# training data using cross-validation.
def best_model(classifier_families, train_instances, judgements):
    best_quality = 0.0
    best_classifier = None
    classifiers = []
    for name, model, parameters in classifier_families:
        classifiers.append(best_config(model, parameters,
                                       train_instances,
                                       judgements))
 
    for name, quality, classifier in classifiers:
        log_info('Considering classifier... ' + name)
        if (quality > best_quality):
            best_quality = quality
            best_classifier = [name, classifier]
 
    log_info('Best classifier... ' + best_classifier[0])
    return best_classifier[1]
 
# List of candidate family classifiers with parameters for grid
# search [name, classifier object, parameters].
def candidate_families():
    candidates = []
    svm_tuned_parameters = [{'kernel': ['poly'],
                             'degree': [1, 2, 3, 4]}]
    candidates.append(["SVM", SVC(C=1), svm_tuned_parameters])
 
    rf_tuned_parameters = [{"n_estimators"      : [200, 500, 1000],
           "max_features"      : [3, 4, 5],
           "max_depth"         : [10, 15, 20],
           "min_samples_leaf" : [2, 3, 4]}]
    candidates.append(["RandomForest",
                       RandomForestClassifier(n_jobs=-1),
                       rf_tuned_parameters])
 
    knn_tuned_parameters = [{"n_neighbors": [1, 3, 5, 10, 20]}]
    candidates.append(["kNN", KNeighborsClassifier(),
                       knn_tuned_parameters])
 
    return candidates

#xgboost
#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=.05)
#gbm = gbm.fit(X_train, y_train)
#predicted_1 = gbm.predict(X_val)
#print metrics.accuracy_score(y_val, predicted_1)

#predictions_xgb = gbm.predict(test_features)
#my_solution_xgb = pd.DataFrame(predictions_xgb, PassengerId, columns = ["Survived"])
#my_solution_xgb.to_csv("my_solution_xgb.csv", index_label = ["PassengerId"])
