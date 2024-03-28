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
from sklearn.utils.class_weight import compute_class_weight

import random
import statistics


# def loadData(path):
#     f = open(path, "r")
#     data = csv.reader(f)  # csv format
#     data = np.array(list(data))  # convert to matrix
#     data = np.delete(data, 0, 0)  # delete header
#     data = np.delete(data, 0, 1)  # delete index
#     np.random.shuffle(data)  # shuffle data
#     f.close()
#     trainSet = data[:, :-1]  # training data from 1 -> 100
#     testSet = data[:, -1]  # the others is testing data
#     return trainSet, testSet


# X, y = loadData("data_modified.csv")

ID_dataset = 44

dataset = openml.datasets.get_dataset(
    dataset_id=ID_dataset, download_data=True, download_qualities=True, download_features_meta_data=True)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Đổi giá trị của cột thứ 2 thành số nguyên
label_encoder = LabelEncoder()
X.iloc[:, 1] = label_encoder.fit_transform(X.iloc[:, 1])


# print("X: ", X)
# print("y: ", y)

# # huan luyen 10 lan
temp_acc = []
temp_p = []
temp_r = []
temp_f1 = []
for time in range(1, 11):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        X, y, test_size=1/3, random_state=None)
    # print("lenX = {0}, lenY = {1}".format(len(X_Train), len(X_Test)))
    # rf_model = RandomForestClassifier()

    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_Train), y=Y_Train)
    # # Tạo từ điển trọng số cho mỗi lớp
    # class_weight = dict(zip(np.unique(Y_Train), class_weights))
    # print("cw", class_weight)

    rf_model = RandomForestClassifier()

    rf_model.fit(X_Train, Y_Train)
    # print("x-trin,y-trin", X_Train, Y_Train)

    Y_Pred = rf_model.predict(X_Test)

    # report = classification_report(Y_Test, Y_Pred, zero_division='warn')

    Acc = accuracy_score(Y_Test, Y_Pred)*100
    # average="weighted"
    P = precision_score(Y_Test, Y_Pred, pos_label='1')*100
    R = recall_score(Y_Test, Y_Pred, pos_label='1')*100
    F1 = f1_score(Y_Test, Y_Pred, pos_label='1')*100
    temp_acc.append(Acc)
    temp_p.append(P)
    temp_r.append(R)
    temp_f1.append(F1)
mean_acc = statistics.mean(temp_acc)
mean_p = statistics.mean(temp_p)
mean_r = statistics.mean(temp_r)
mean_f1 = statistics.mean(temp_f1)
print("Trung binh 10 lan: acc= {} , p= {}, r= {}, f1= {}".format(
    round(mean_acc, 2), round(mean_p, 2), round(mean_r, 2), round(mean_f1, 2)))
