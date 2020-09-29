import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from numpy.random import default_rng
data = pd.read_csv("feng_x.csv")
# data = pd.read_csv("ibdfullHS_UCr_x.csv")
result_colName = "RS"
#rename to RS if use old data
colName = data.columns
df = pd.DataFrame(data, columns=colName)
X = df[colName]
y = df[result_colName]
df.head()
# =======Method 1: get 70-30==========
#===> 2. / Get 70% train set from set Feng
# Split dataset into training set and test set
# 70% training and 30% test
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
#===> 3. / Let find its cor
# X_train_new = X_train
# plt.figure(figsize=(15,15))
cor = df.corr()  #tính độ tương quan của dữ liệu trong df
# sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
# plt.show()
#Correlation with output variable
cor_target = abs(
    cor[result_colName])  #lấy trị tuyệt đối của độ tương quan dữ liệu
#Selecting highly correlated features
relevant_features = cor_target[
    cor_target > 0.1]  #lọc những thằng có độ tương quan lớn hơn 0.1
if len(relevant_features.index) <= 1:
	print("More than one importance feature is required")
	quit()
else:
	print("Found " + str(len(relevant_features.index)) + " important features")
importanceFeature = relevant_features.index  #index là tên cột, values là giá trị của cột
#print(relevant_features.index)
# print('#importance features')
#print(importanceFeature)#importance feature
# =======Method 2: get random new feature equal important-feature==========
rng = default_rng()  #random theo chuẩn quy định python
# random from original feng with size columns equal newColumNameFeng (important-feature)
numbers = rng.choice(
    len(colName) - 1, size=len(importanceFeature) - 1, replace=False
)  #Random toàn bộ feature ban đầu (bỏ cột RS) và trả ra numbers feature
# Set new size col for Feng: Get original feng delete (patients name), then set size equal numbers plus RS column
randomeFeatureSameSize = colName.delete(0).take(numbers).append(
    pd.Index([result_colName]))  #bỏ cột victim ban đầu và lấy
#print(colName.delete(0).take(numbers))
#print(randomeFeatureSameSize)#random features
# set new data after random
X_Train_Random = df[randomeFeatureSameSize].drop(
    result_colName, 1)  #lấy hết randomeFeatureSameSize cột từ df và bỏ cột RS
y_Train_Random = df[result_colName]  # Labels
#===> 4. / Reconstruct dataframe Feng with already 70% train and cor >0.1
X_Train_ImportFeature = df[importanceFeature].drop(result_colName, 1)
y_Train_ImportFeature = df[result_colName]  # Labels
fileList = []
# fileList.append('yu_x.csv')
# fileList.append('zeller_x.csv')
# fileList.append('vogtmann_x.csv')
nTimesRun = 2
fileList.append('ibdfullHS_UCr_x.csv')
fileList.append('ibdfullHS_CDr_x.csv')
fileList.append('ibdfullHS_iCDf_x.csv')
fileList.append('ibdfullHS_iCDr_x.csv')
fileList.append('ibdfullHS_UCf_x.csv')
acc_if = 0.0
mcc_if = 0.0
auc_if = 0.0
acc_random = 0.0
mcc_random = 0.0
auc_random = 0.0
#This for test of another .csv files.
for x in range(len(fileList)):  #run fileList size times
  for n in range(nTimesRun):  #tự gán
    if nTimesRun ==0:
	    break
    print(fileList[x] + " Run time " + str(n))
    data_yu = pd.read_csv(fileList[x])
    df_IF = pd.DataFrame(data_yu, columns=importanceFeature).fillna(0)
    X_Test_IF = df_IF[importanceFeature].drop(result_colName, 1)
    y_Test_IF = df_IF[result_colName]  # Labels

    # Remove feature and Test with method 2 - random
    df_Test = pd.DataFrame(data_yu, columns=randomeFeatureSameSize).fillna(0)
    X_Test_Random = df_Test[randomeFeatureSameSize].drop(result_colName, 1)
    y_Test_Random = df_Test[result_colName]  # Labels

    # Train with method 2
    clfRandom = RandomForestClassifier(n_estimators=1000, max_features='auto')  #config train
    clfRandom.fit(X_Train_Random, y_Train_Random) #train
    y_Pred_Random = clfRandom.predict(X_Test_Random)

    acc_random += metrics.accuracy_score(y_Test_Random.round(), y_Pred_Random.round())
    mcc_random += metrics.matthews_corrcoef(y_Test_Random.round(),y_Pred_Random.round())
    auc_random += metrics.roc_auc_score(y_Test_Random.round(),y_Pred_Random.round())

    # Train with method 1
    clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
    clf.fit(X_Train_ImportFeature, y_Train_ImportFeature)  #Build a forest of trees from the training set (X, y).
    y_Predict_IF = clf.predict(X_Test_IF)

    acc_if += metrics.accuracy_score(y_Test_IF.round(),y_Predict_IF.round())
    mcc_if += metrics.matthews_corrcoef(y_Test_IF.round(),y_Predict_IF.round())
    auc_if += metrics.roc_auc_score(y_Test_IF.round(),y_Predict_IF.round())
  if nTimesRun ==0:
    break
  print("====== Random run " + str(nTimesRun) + " times =====")
  print("Average ACC = " + str(acc_random / nTimesRun))
  print("Average MCC = " + str(mcc_random / nTimesRun))
  print("Average AUC = " + str(auc_random / nTimesRun))
  print("====== Importance Feature run " + str(nTimesRun) + " times =====")
  print("Average ACC = " + str(acc_if / nTimesRun))
  print("Average MCC = " + str(mcc_if / nTimesRun))
  print("Average AUC = " + str(auc_if / nTimesRun))
  acc_if = 0.0
  mcc_if = 0.0
  auc_if = 0.0
  acc_random = 0.0
  mcc_random = 0.0
  auc_random = 0.0
  print("++++++++++++++++++++")
