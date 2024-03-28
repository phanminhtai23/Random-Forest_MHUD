from sklearn.preprocessing import LabelEncoder
import openml
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report



import random
import statistics


def loadData(path):
    f = open(path, "r")
    data = csv.reader(f)  # csv format
    data = np.array(list(data))  # convert to matrix
    data = np.delete(data, 0, 0)  # delete header
    data = np.delete(data, 0, 1)  # delete index
    np.random.shuffle(data)  # shuffle data
    f.close()
    trainSet = data[:, :-1]  # training data from 1 -> 100
    testSet = data[:, -1]  # the others is testing data
    return trainSet, testSet


X, y = loadData("data_modified.csv")

# print("X: ", X)
# print("y: ", y)

param_grid = [
    {'n_estimators': 60, 'max_features': None, 'max_depth': 2},
    {'n_estimators': 60, 'max_features': None, 'max_depth': 12},
    {'n_estimators': 60, 'max_features': None, 'max_depth': 15},
    {'n_estimators': 60, 'max_features': 'sqrt', 'max_depth': 2},
    {'n_estimators': 60, 'max_features': 'sqrt', 'max_depth': 12},
    {'n_estimators': 60, 'max_features': 'sqrt', 'max_depth': 15},
    {'n_estimators': 60, 'max_features': 'log2', 'max_depth': 2},
    {'n_estimators': 60, 'max_features': 'log2', 'max_depth': 12},
    {'n_estimators': 60, 'max_features': 'log2', 'max_depth': 15},


    {'n_estimators': 110, 'max_features': None, 'max_depth': 2},
    {'n_estimators': 110, 'max_features': None, 'max_depth': 12},
    {'n_estimators': 110, 'max_features': None, 'max_depth': 15},
    {'n_estimators': 110, 'max_features': 'sqrt', 'max_depth': 2},
    {'n_estimators': 110, 'max_features': 'sqrt', 'max_depth': 12},
    {'n_estimators': 110, 'max_features': 'sqrt', 'max_depth': 15},
    {'n_estimators': 110, 'max_features': 'log2', 'max_depth': 2},
    {'n_estimators': 110, 'max_features': 'log2', 'max_depth': 12},
    {'n_estimators': 110, 'max_features': 'log2', 'max_depth': 15},

    {'n_estimators': 200, 'max_features': None, 'max_depth': 2},
    {'n_estimators': 200, 'max_features': None, 'max_depth': 12},
    {'n_estimators': 200, 'max_features': None, 'max_depth': 15},
    {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 2},
    {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 12},
    {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 15},
    {'n_estimators': 200, 'max_features': 'log2', 'max_depth': 2},
    {'n_estimators': 200, 'max_features': 'log2', 'max_depth': 12},
    {'n_estimators': 200, 'max_features': 'log2', 'max_depth': 15}
    # Thêm các bộ tham số khác vào đây nếu cần
]
param_grid1 = [
    {'max_depth': 1},
    {'max_depth': 2},
    {'max_depth': 3},
    {'max_depth': 4},
    {'max_depth': 5},
    {'max_depth': 6},
    {'max_depth': 7},
    {'max_depth': 8},
    {'max_depth': 9},
    {'max_depth': 10},
    {'max_depth': 11},
    {'max_depth': 12},
    {'max_depth': 13},
    {'max_depth': 14},
    {'max_depth': 15},
    {'max_depth': 16},
    {'max_depth': 17},
    {'max_depth': 18},
    {'max_depth': 19},
    {'max_depth': 20},
]
param_grid2 = [
    {'n_estimators': 50},
    {'n_estimators': 60},
    {'n_estimators': 70},
    {'n_estimators': 80},
    {'n_estimators': 90},
    {'n_estimators': 100},
    {'n_estimators': 110},
    {'n_estimators': 120},
    {'n_estimators': 130},
    {'n_estimators': 140},
    {'n_estimators': 150},
    {'n_estimators': 160},
    {'n_estimators': 170},
    {'n_estimators': 180},
    {'n_estimators': 190},
    {'n_estimators': 200},
    {'n_estimators': 250},
    {'n_estimators': 300},
    {'n_estimators': 350},
    {'n_estimators': 400},
]
param_grid3 = [
    {'max_features': None},
    {'max_features': 'sqrt'},
    {'max_features': 'log2'}
]
param_grid4 = [
    {'bootstrap': True},
    {'bootstrap': False},
]
param_grid5 = [
    {'max_features': None},
    {'max_features': 'sqrt'},
    {'max_features': 'log2'}
]
# # huan luyen 10 lan
# temp_acc = []
# temp_p = []
# temp_r = []
# temp_f1 = []

best_score = 0
best_params = None

for params in param_grid:
    temp_acc = []
    temp_p = []
    temp_r = []
    temp_f1 = []
    for time in range(1, 11):
        combined = list(zip(X, y))
        random.shuffle(combined)
        dulieu_X_shuffled, dulieu_Y_shuffled = zip(*combined)
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(
            dulieu_X_shuffled, dulieu_Y_shuffled, test_size=1/3, random_state=42)
        # print("lenX = {0}, lenY = {1}".format(len(X_Train), len(X_Test)))
        # rf_model = RandomForestClassifier()
        rf_model = RandomForestClassifier()

        rf_model.set_params(**params)
        rf_model.fit(X_Train, Y_Train)

        Y_Pred = rf_model.predict(X_Test)

        report = classification_report(Y_Test, Y_Pred, zero_division=0.0)

        Acc = accuracy_score(Y_Pred, Y_Test)*100
        P = precision_score(Y_Pred, Y_Test, average="weighted")*100
        R = recall_score(Y_Pred, Y_Test, average="weighted")*100
        F1 = f1_score(Y_Pred, Y_Test, average="weighted")*100
        temp_acc.append(Acc)
        temp_p.append(P)
        temp_r.append(R)
        temp_f1.append(F1)
    mean_acc = statistics.mean(temp_acc)
    mean_p = statistics.mean(temp_p)
    mean_r = statistics.mean(temp_r)
    mean_f1 = statistics.mean(temp_f1)
    print("Parameter:", params)
    print("Trung binh 10 lan: acc= {} , p= {}, r= {}, f1= {}".format(
        round(mean_acc, 2), round(mean_p, 2), round(mean_r, 2), round(mean_f1, 2)))

    # if score > best_score:
    #     best_score = score
    #     best_params = params

    # print("Best parameters found: ", best_params)
    # print("Best accuracy on test set: ", best_score)
