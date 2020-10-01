import pandas as pd
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold


#Revise is needed
# Train on one dataset, then test of another dataset
def correction_kfold_selftest(filenameTrain, resultColName, nTimes, percentTest, coef_percent, numK):
    print("Train by " + str(filenameTrain) + ", run seft - test " + str(nTimes) + " times, use " + str(
        percentTest) + " percent from train to get coef and coef > " + str(coef_percent))

    data = pd.read_csv(filenameTrain)
    colName = data.columns;
    df = pd.DataFrame(data, columns=colName)
    X = df[colName]
    y = df[resultColName]
    df.head()
    if percentTest > 0.0 and percentTest < 1.0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentTest)
        X_train_new = X_train
    else:
        X_train_new = X
        print("You are select a whole set for training!")
    print("Get " + str(len(X_train_new)) + " victims from " + str(len(X)) + " victims for get Coefficient.")
    if float(coef_percent) != 0.0:
        cor = X_train_new.corr()
        cor_target = abs(cor[resultColName])
        relevant_features = cor_target[cor_target > float(coef_percent)]
    else:
        relevant_features = X_train_new
    if len(relevant_features.index) <= 1:
        print("More than one importance feature is required")
        quit()
    else:
        print("Found " + str(len(relevant_features.index)) + " important features")
    importanceFeature = relevant_features.index
    rng = default_rng()
    # In colName has n columns, position of RS is n - 1. Because of a noname rows of V1,V2,V3,...
    numbers = rng.choice(len(colName) - 2, size=len(importanceFeature) - 1, replace=False)
    randomeFeatureSameSize = colName.delete(0).take(numbers).append(pd.Index([resultColName]))
    X_Train_Random = df[randomeFeatureSameSize].drop(resultColName, 1)
    y_Train_Random = df[resultColName]
    X_Train_ImportFeature = df[importanceFeature].drop(resultColName, 1)
    y_Train_ImportFeature = df[resultColName]
    acc_random = 0.0
    mcc_random = 0.0
    auc_random = 0.0
    acc_if = 0.0
    mcc_if = 0.0
    auc_if = 0.0
    for n in range(nTimes):
        if nTimes == 0:
            break
        print("Self-Test - Time run number " + str(n + 1))
        df_IF = pd.DataFrame(data, columns=importanceFeature).fillna(0)
        X_Test_IF = df_IF[importanceFeature].drop(resultColName, 1)
        y_Test_IF = df_IF[resultColName]  # Labels

        # Remove feature and Test with method 2 - random
        df_Test = pd.DataFrame(data, columns=randomeFeatureSameSize).fillna(0)
        X_Test_Random = df_Test[randomeFeatureSameSize].drop(resultColName, 1)
        y_Test_Random = df_Test[resultColName]  # Labels

        # Train with method 2
        clfRandom = RandomForestClassifier(n_estimators=1000, max_features='auto')
        clfRandom.fit(X_Train_Random, y_Train_Random)
        y_Pred_Random = clfRandom.predict(X_Test_Random)
        acc_random += metrics.accuracy_score(y_Test_Random, y_Pred_Random.round())
        mcc_random += metrics.matthews_corrcoef(y_Test_Random, y_Pred_Random.round())
        auc_random += metrics.roc_auc_score(y_Test_Random, y_Pred_Random.round())

        # Train with method 1: random forest and kFold
        clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
        kf = KFold(n_splits=int(numK))
        X_new_kfold = X_Train_ImportFeature
        y_new_kfold = y_Train_ImportFeature
        accuracy_model_acc = []
        accuracy_model_mcc = []
        accuracy_model_auc = []
        for train, test in kf.split(X_new_kfold):
            X_train_kfold_loops, X_test_kfold_loops = X_new_kfold.iloc[train], X_new_kfold.iloc[test]
            y_train_kfold_loops, y_test_kfold_loops = y_new_kfold[train], y_new_kfold[test]

            # Train the model
            model = clf.fit(X_train_kfold_loops, y_train_kfold_loops)
            accuracy_model_acc.append(metrics.accuracy_score(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
            accuracy_model_mcc.append(
                metrics.matthews_corrcoef(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
            accuracy_model_auc.append(metrics.roc_auc_score(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
        clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
        clf.fit(X_Train_ImportFeature,
                y_Train_ImportFeature)  # Build a forest of trees from the training set (X, y).
        y_Predict_IF = clf.predict(X_test_kfold_loops)

        acc_if += metrics.accuracy_score(y_test_kfold_loops, y_Predict_IF)
        mcc_if += metrics.matthews_corrcoef(y_test_kfold_loops, y_Predict_IF)
        auc_if += metrics.roc_auc_score(y_test_kfold_loops, y_Predict_IF)

        # Train with method 2: Random
        clfRandom = RandomForestClassifier(n_estimators=1000, max_features='auto')
        clfRandom.fit(X_Train_Random, y_Train_Random)
        y_Pred_Random = clfRandom.predict(X_Test_Random)
        accuracy_model_acc_random = []
        accuracy_model_mcc_random = []
        accuracy_model_auc_random = []

        for train, test in kf.split(X_new_kfold):
            X_train_kfold_loops, X_test_kfold_loops = X_new_kfold.iloc[train], X_new_kfold.iloc[test]
            y_train_kfold_loops, y_test_kfold_loops = y_new_kfold[train], y_new_kfold[test]

            # Train the model
            model = clf.fit(X_train_kfold_loops, y_train_kfold_loops)
            accuracy_model_acc_random.append(
                metrics.accuracy_score(y_Test_Random, model.predict(X_Test_Random)))
            accuracy_model_mcc_random.append(
                metrics.matthews_corrcoef(y_Test_Random, model.predict(X_Test_Random)))
            accuracy_model_auc_random.append(
                metrics.roc_auc_score(y_Test_Random, model.predict(X_Test_Random)))

        acc_if += float(sum(map(float, accuracy_model_acc)) / len(accuracy_model_acc))
        mcc_if += float(sum(map(float, accuracy_model_mcc)) / len(accuracy_model_mcc))
        auc_if += float(sum(map(float, accuracy_model_auc)) / len(accuracy_model_auc))
        acc_random += float(sum(map(float, accuracy_model_acc_random)) / len(accuracy_model_acc_random))
        mcc_random += float(sum(map(float, accuracy_model_mcc_random)) / len(accuracy_model_mcc_random))
        auc_random += float(sum(map(float, accuracy_model_auc_random)) / len(accuracy_model_auc_random))
    print("Self-test Result ")

    print("Random run " + str(nTimes) + " times =====")
    print("Average ACC = " + str(acc_random / nTimes))
    print("Average MCC = " + str(mcc_random / nTimes))
    print("Average MCC = " + str(auc_random / nTimes))
    print("Importance Feature run " + str(nTimes) + " times =====")
    print("Average ACC = " + str(acc_if / nTimes))
    print("Average MCC = " + str(mcc_if / nTimes))
    print("Average MCC = " + str(auc_if / nTimes))
    acc_if = 0.0
    mcc_if = 0.0
    auc_if = 0.0
    acc_random = 0.0
    mcc_random = 0.0
    auc_random = 0.0
