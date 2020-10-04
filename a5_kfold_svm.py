import pandas as pd
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from a_utils import *
import pandas as pd
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import linear_model
from a_utils import *
from sklearn import feature_selection
import statsmodels.api as sm
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
def kfold_svm(filenameTrain, resultColName, fileListTest, nTimes, coef_percent, numK, flag, nlargestFeatures):
    importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature = findImportancesFeatures(resultColName,
                                                                                              filenameTrain,
                                                                                              coef_percent, flag,
                                                                                              nlargestFeatures)
    randomeFeatureSameSize, X_Train_Random, y_Train_Random = findRandomeFeaturesSets(resultColName, filenameTrain,
                                                                                     len(importanceFeature))
    print("KFold = " + str(numK))
    acc_random = 0.0
    mcc_random = 0.0
    auc_random = 0.0
    acc_if = 0.0
    mcc_if = 0.0
    auc_if = 0.0
    print("Bắt đầu kết quả ----------------- ")
    for x in range(len(fileListTest)):
        print("Chạy test trên " + fileListTest[x] + ". Chạy lặp " + str(nTimes) + " lần.")
        for n in range(nTimes):
            if nTimes == 0:
                break
            data_yu = pd.read_csv(fileListTest[x])
            df_IF = pd.DataFrame(data_yu, columns=importanceFeature).fillna(0)
            X_Test_IF = df_IF[importanceFeature]
            y_Test_IF = data_yu[resultColName]  # Labels
            # Remove feature and Test with method 2 - random
            df_Test = pd.DataFrame(data_yu, columns=randomeFeatureSameSize).fillna(0)
            X_Test_Random = df_Test[randomeFeatureSameSize]
            y_Test_Random = data_yu[resultColName]  # Labels

            # Train with method 1: random forest and kFold
            kf = KFold(n_splits=int(numK),random_state=42, shuffle=True)
            X_new_kfold = X_Train_ImportFeature
            y_new_kfold = y_Train_ImportFeature
            accuracy_model_acc = []
            accuracy_model_mcc = []
            accuracy_model_auc = []
            select = SelectKBest(mutual_info_classif, k=int(numK))
            scl = StandardScaler()
            svm = SVC(kernel='linear', probability=True, random_state=42)
            pipeline = Pipeline([('select', select), ('scale', scl), ('svm', svm)])
            for train, test in kf.split(X_new_kfold):
                X_train_kfold_loops, X_test_kfold_loops = X_new_kfold.iloc[train], X_new_kfold.iloc[test]
                y_train_kfold_loops, y_test_kfold_loops = y_new_kfold[train], y_new_kfold[test]

                # Train the model
                model = pipeline.fit(X_train_kfold_loops, y_train_kfold_loops)
                accuracy_model_acc.append(metrics.accuracy_score(y_Test_IF, model.predict(X_Test_IF).round()))
                accuracy_model_mcc.append(
                    metrics.matthews_corrcoef(y_Test_IF, model.predict(X_Test_IF).round()))
                accuracy_model_auc.append(metrics.roc_auc_score(y_Test_IF, model.predict(X_Test_IF).round()))
            X_new_kfold = X_Train_Random
            y_new_kfold = y_Train_Random
            accuracy_model_acc_random = []
            accuracy_model_mcc_random = []
            accuracy_model_auc_random = []

            select = SelectKBest(mutual_info_classif, k=int(numK))
            scl = StandardScaler()
            svm = SVC(kernel='linear', probability=True, random_state=42)
            pipeline = Pipeline([('select', select), ('scale', scl), ('svm', svm)])
            for train, test in kf.split(X_new_kfold):
                X_train_kfold_loops, X_test_kfold_loops = X_new_kfold.iloc[train], X_new_kfold.iloc[test]
                y_train_kfold_loops, y_test_kfold_loops = y_new_kfold[train], y_new_kfold[test]

                # Train the model
                model = pipeline.fit(X_train_kfold_loops, y_train_kfold_loops)
                accuracy_model_acc_random.append(
                    metrics.accuracy_score(y_Test_Random, model.predict(X_Test_Random).round()))
                accuracy_model_mcc_random.append(
                    metrics.matthews_corrcoef(y_Test_Random, model.predict(X_Test_Random).round()))
                accuracy_model_auc_random.append(
                    metrics.roc_auc_score(y_Test_Random, model.predict(X_Test_Random).round()))

            acc_if += sumThenAveragePercisely(accuracy_model_acc)
            mcc_if += sumThenAveragePercisely(accuracy_model_mcc)
            auc_if += sumThenAveragePercisely(accuracy_model_auc)
            acc_random += sumThenAveragePercisely(accuracy_model_acc_random)
            mcc_random += sumThenAveragePercisely(accuracy_model_mcc_random)
            auc_random += sumThenAveragePercisely(accuracy_model_auc_random)
            if nTimes == 0:
                break
        printResult(acc_random, mcc_random, auc_random, acc_if, mcc_if, auc_if, nTimes)
        acc_if = 0.0
        mcc_if = 0.0
        auc_if = 0.0
        acc_random = 0.0
        mcc_random = 0.0
        auc_random = 0.0

