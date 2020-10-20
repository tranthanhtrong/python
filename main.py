from a_utils import *
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt

#Lựa data
print("Lựa data, 1 cho bộ cũ (feng, vogtmann, yu, zeller), 2 cho bộ mới (bộ IBD):")
dataset_choice = input("Lựa : ")
dataset_choice = int(dataset_choice)
if (dataset_choice == 1):
    team_file = getOldDataset()
elif (dataset_choice == 2):
    team_file = getNewDataset()

#Đọc data
data = pd.read_csv(team_file.train)
colName = data.columns
df = pd.DataFrame(data, columns=colName)
df.head()
X = df[colName]
y = df[team_file.resultColName]
columns = X.columns.tolist()
X_new = X.drop(X.columns[[0, len(columns)-1]], axis=1)
data_top = X_new.columns.values

#Trích đặc trưng
clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
clf.fit(X_new, y)

#Lọc 100 đặc trưng từ cao xuống thấp
sorted_idx = (-clf.feature_importances_).argsort(-1)[:100]
features_value = clf.feature_importances_[sorted_idx]
new_name = data_top[sorted_idx]

nTimes = input("Số lần lặp tăng feature: ")
nTimes = int(nTimes)
# nTimes = 100

acc_if = 0.0
print("Bắt đầu kết quả ----------------- ")
acc_average = []
for x in range(len(team_file.listFileTest)):
    print("======Chạy test trên " + team_file.listFileTest[x] + "======")
    for n in range(nTimes):
        if nTimes == 0:
            break
        data_yu = pd.read_csv(team_file.listFileTest[1])
        importanceFeature = new_name[:(n+1)]
        X_train_IF = df[importanceFeature]
        y_train_IF = y
        df_IF = pd.DataFrame(data_yu, columns=importanceFeature).fillna(0)
        X_Test_IF = df_IF[importanceFeature]
        y_Test_IF = data_yu[team_file.resultColName]

        clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
        clf.fit(X_train_IF, y_train_IF)  # Build a forest of trees from the training set (X, y).
        y_Predict_IF = clf.predict(X_Test_IF)

        acc_if += metrics.accuracy_score(y_Test_IF, y_Predict_IF.round())
        acc_average.append(acc_if)
        print("Chạy lặp " + str(n + 1) + " feature: " + str(acc_if))
        if nTimes == 0:
            break
        acc_if = 0.0
# acc_if = 0.0
average = sum(acc_average)/len(acc_average)
print("Average ACC:")
print(average)
