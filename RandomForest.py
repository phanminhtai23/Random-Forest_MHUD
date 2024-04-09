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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import random
import statistics

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
    combined = list(zip(X.values, y))
    random.shuffle(combined)
    dulieu_X_shuffled, dulieu_Y_shuffled = zip(*combined)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        dulieu_X_shuffled, dulieu_Y_shuffled, test_size=1/3, random_state=42, shuffle=True)

    rf_model = RandomForestClassifier()

    rf_model.fit(X_Train, Y_Train)

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

labels = ['1', '2']
cm = confusion_matrix(Y_Test, Y_Pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
