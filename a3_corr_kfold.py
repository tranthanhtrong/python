import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
from numpy import std
import seaborn as sns
import numpy as np
from sklearn import metrics, pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate
from numpy.random import default_rng
from a_utils import *

#Train on one dataset, then test of another dataset
def correction_kfold(filenameTrain,resultColName,fileListTest,nTimes,percentTest,coef_percent, numK):
  print("Train by " + str(filenameTrain) +", run test " + str(nTimes) + " times, use " + str(percentTest) +" percent from train to get coef and coef > " + str(coef_percent))

  data = pd.read_csv(filenameTrain)
  colName = data.columns;
  df = pd.DataFrame(data, columns = colName)
  X=df[colName]
  y=df[resultColName]
  df.head()
  if percentTest>0.0 and percentTest < 1.0:
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=percentTest)
    X_train_new = X_train
  else :
    X_train_new = X
  print("Get " + str(len(X_train_new)) + " columns from " + str(len(X)) + " elements for get Coefficient.")
  if float(coef_percent)!=0.0:
    cor = X_train_new.corr()
    cor_target = abs(cor[resultColName])
    relevant_features = cor_target[cor_target > float(coef_percent)]
  else:
    relevant_features = X_train_new
  if len(relevant_features.index) <= 1:
    print("More than one importance feature is required")
    quit()
  else :
    print("Found " + str(len(relevant_features.index)) + " important features")
  importanceFeature =  relevant_features.index;
  rng = default_rng()
  #In colName has n columns, position of RS is n - 1. Because of a noname rows of V1,V2,V3,...
  numbers = rng.choice(len(colName)-2, size=len(importanceFeature)-1, replace=False)
  randomeFeatureSameSize = colName.delete(0).take(numbers).append(pd.Index([resultColName]))
  X_Train_Random = df[randomeFeatureSameSize].drop(resultColName,1)
  y_Train_Random = df[resultColName]
  X_Train_ImportFeature=df[importanceFeature].drop(resultColName,1)
  y_Train_ImportFeature=df[resultColName]
  acc_random =0.0
  mcc_random =0.0
  auc_random =0.0
  acc_if =0.0
  mcc_if =0.0
  auc_if =0.0
  for x in range(len(fileListTest)):
    print("Test on " + fileListTest[x] )
    for n in range(nTimes):
        if nTimes ==0:
          break
        print("Time run number " + str(n+1))
        data_yu = pd.read_csv(fileListTest[x])
        df_IF = pd.DataFrame(data_yu, columns = importanceFeature).fillna(0)
        X_Test_IF=df_IF[importanceFeature].drop(resultColName,1)
        y_Test_IF=df_IF[resultColName] # Labels

        # Remove feature and Test with method 2 - random
        df_Test = pd.DataFrame(data_yu, columns = randomeFeatureSameSize).fillna(0)
        X_Test_Random=df_Test[randomeFeatureSameSize].drop(resultColName,1)
        y_Test_Random=df_Test[resultColName] # Labels

        # Train with method 2: Random
        clfRandom=RandomForestClassifier(n_estimators=1000,max_features='auto')
        clfRandom.fit(X_Train_Random,y_Train_Random)
        y_Pred_Random=clfRandom.predict(X_Test_Random)

        acc_random+=metrics.accuracy_score(y_Test_Random, y_Pred_Random.round())
        mcc_random+=metrics.matthews_corrcoef(y_Test_Random, y_Pred_Random.round())
        auc_random+=metrics.roc_auc_score(y_Test_Random, y_Pred_Random.round())

        # Train with method 1: random forest and kFold
        clf=RandomForestClassifier(n_estimators=1000,max_features='auto')
        kf = KFold(n_splits=int(numK))
        # enumerate splits
        output = cross_validate(clf, X_Train_ImportFeature, y_Train_ImportFeature, cv=kf)
        print("===========")
        print(output)
        # for train_ix, test_ix in kf.split(X_Train_ImportFeature):
        #     # get data
        #     train_X, test_X = X_Train_ImportFeature[train_ix], X_Train_ImportFeature[test_ix]
        #     train_y, test_y = y_Train_ImportFeature[train_ix], y_Train_ImportFeature[test_ix]
        #     print(train_X)
        #     print(train_y)
        #     print("=======")
        #     print(test_X)
        #     print(test_y)
            # fit model
            # clf.fit(train, y_Train_ImportFeature)
            # evaluate model
            # yhat = model.predict(test_X)
            # acc = accuracy_score(test_y, yhat)
        # output = cross_validate(clf, X_Train_ImportFeature, y_Train_ImportFeature, cv=kf, scoring = ('accuracy', 'f1', 'roc_auc'), return_estimator = True, return_train_score= True)
        # result = next(kf.split(X_Train_ImportFeature, y_Train_ImportFeature), None)
        # result_column = next(kf.split(y_Train_ImportFeature), None)
        # output = cross_validate(clf, X, y, scoring='accuracy', cv=5, n_jobs=-1)
        # train = X_Train_ImportFeature.iloc[result[0]]
        # clf.fit(train,y_Train_ImportFeature) #Build a forest of trees from the training set (X, y).
        # y_Predict_IF=clf.predict(X_Test_IF)
        # print("===============")
        # kf = KFold(n_splits=5, shuffle=True, random_state=2)
        # result = next(kf.split(X_Train_ImportFeature), None)
        # # print(result)
        # print(train)
        print("++++++++")
        # print(y_Train_ImportFeature)

        # test = X_Train_ImportFeature.iloc[result[1]]
        # print(X_Train_ImportFeature)
        print("------")
        # print(train_X)
        print("++++++++")
        # print(train_y)
        # print(std(output))


        # acc_if+=metrics.accuracy_score(y_Test_IF, y_Predict_IF.round())
        # mcc_if+=metrics.matthews_corrcoef(y_Test_IF, y_Predict_IF.round())
        # auc_if+=metrics.roc_auc_score(y_Test_IF, y_Predict_IF.round())
    print("Result ")
    if nTimes ==0:
      break
    # print("Random run " + str(nTimes) + " times =====")
    # print("Average ACC = "  + str(acc_random/nTimes))
    # print("Average MCC = "  + str(mcc_random/nTimes))
    # print("Average MCC = "  + str(auc_random/nTimes))
    # print("Importance Feature run " + str(nTimes) + " times =====")
    # print("Average ACC = "  + str(acc_if/nTimes))
    # print("Average MCC = "  + str(mcc_if/nTimes))
    # print("Average MCC = "  + str(auc_if/nTimes))
    acc_if =0.0
    mcc_if =0.0
    auc_if =0.0
    acc_random =0.0
    mcc_random =0.0
    auc_random =0.0