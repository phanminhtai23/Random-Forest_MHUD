# chuyển cột giới tình thành số (1: nam, 0: nữ)

import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv("data.csv")

# Thay đổi các giá trị "Male" và "Female" thành 1 và 0
df['V2'].replace({'Male': 1, 'Female': 0}, inplace=True)

# Lưu lại file CSV đã thay đổi
df.to_csv("data_modified.csv", index=False)
