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


# In ra thông tin về tập dữ liệu
print(f"ID tập dữ liệu: {dataset.dataset_id}")

print(f"Tên tập dữ liệu: {dataset.name}")

# print(f"Mô tả: {dataset.description}")

print(f"Số lượng thuộc tính: {dataset.qualities['NumberOfFeatures']}")

print(f"Số lượng mẫu: {dataset.qualities['NumberOfInstances']}")

# print(f"Số lượng lớp: {len(dataset.retrieve_class_labels())}")



# Lấy dữ liệu và nhãn
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
# df.to_csv('data.csv', index=False)

# print(X.dtypes, y.dtype)

# # In ra 5 dòng đầu tiên của dữ liệu
# print("Dữ liệu (5 dòng đầu):")
# print(X.head())

# # In ra 5 dòng đầu tiên của nhãn
# print("Nhãn (5 dòng đầu):")
# print(y.head())

# Đếm số lượng mẫu cho từng lớp
unique_classes, counts = np.unique(y, return_counts=True)

# Vẽ biểu đồ cột
plt.bar(unique_classes, counts, tick_label=unique_classes, color='skyblue')
plt.xlabel('Lớp')
plt.ylabel('Số lượng')
plt.title('Biểu đồ số lượng mẫu cho từng lớp trong dữ liệu Iris')
plt.show()

