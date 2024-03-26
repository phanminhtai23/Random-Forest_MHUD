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

import random
import statistics

# Tải tập dữ liệu Iris (ID=1464) từ OpenML
dataset_id = 1480
dataset = openml.datasets.get_dataset(dataset_id)

# Lấy dữ liệu và nhãn
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

print(f"Số lượng mẫu: {dataset.qualities['NumberOfInstances']}")
# huan luyen 10 lan
temp_acc= []
temp_p= []
temp_r= []
temp_f1= []
for time in range (1,11):
  combined = list(zip(X, y))
  random.shuffle(combined)
  dulieu_X_shuffled, dulieu_Y_shuffled =zip(*combined)
  X_Train, X_Test, Y_Train, Y_Test = train_test_split(dulieu_X_shuffled,dulieu_Y_shuffled, test_size=1/3, random_state=42)
#   print("lenX = {0}, lenY = {}".format(len(X_Train), len(Y_Train)))
  rf_model = RandomForestClassifier(class_weight='balanced')
  rf_model.fit(X_Train, Y_Train)

  Y_Pred =  rf_model.predict(X_Test)
  Acc = accuracy_score(Y_Pred,Y_Test)*100
  P = precision_score(Y_Pred, Y_Test, average="weighted")*100
  R = recall_score(Y_Pred, Y_Test, average="weighted")*100
  F1= f1_score(Y_Pred, Y_Test, average="weighted")*100
  temp_acc.append(Acc)
  temp_p.append(P)
  temp_r.append(R)
  temp_f1.append(F1)
mean_acc = statistics.mean(temp_acc)
mean_p = statistics.mean(temp_p)
mean_r = statistics.mean(temp_r)
mean_f1 = statistics.mean(temp_f1)
print("Trung binh 10 lan: acc={} , p={}, r={}, f1={}".format(round(mean_acc, 2), round(mean_p, 2), round(mean_r, 2), round(mean_f1, 2)))
