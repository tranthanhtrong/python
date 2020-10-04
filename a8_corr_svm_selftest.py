import pandas as pd
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

def correction_svm_selftest(filenameTrain, resultColName, nTimes, coef_percent,flag,nlargestFeatures):
    importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature = findImportancesFeatures(resultColName,
                                                                                              filenameTrain,
                                                                                              coef_percent, flag,
                                                                                              nlargestFeatures)
    randomeFeatureSameSize, X_Train_Random, y_Train_Random = findRandomeFeaturesSets(resultColName, filenameTrain,
                                                                                     len(importanceFeature))
    X_train_IF_Div, X_test_IF_Div, y_train_IF_Div, y_test_IF_Div = train_test_split(X_Train_ImportFeature,
                                                                                    y_Train_ImportFeature,
                                                                                    test_size=0.3)
    X_train_Random_Div, X_test_Random_Div, y_Train_Random_Div, y_test_Random_Div = train_test_split(X_Train_Random,
                                                                                                    y_Train_Random,
                                                                                                    test_size=0.3)
    print("Tự kiểm tra trên 0.3 phần trăm bệnh nhân còn lại. Chạy lặp " + str(nTimes) + " lần.")
    acc_if = 0.0
    mcc_if = 0.0
    auc_if = 0.0
    acc_random = 0.0
    mcc_random = 0.0
    auc_random = 0.0
    for n in range(nTimes):
        if nTimes == 0:
            break
        df_IF = pd.DataFrame(X_test_IF_Div, columns=importanceFeature).fillna(0)
        X_Test_IF = df_IF[importanceFeature]
        y_Test_IF = y_test_IF_Div

        # Remove feature and Test with method 2 - random
        df_Test = pd.DataFrame(X_test_Random_Div, columns=randomeFeatureSameSize).fillna(0)
        X_Test_Random = df_Test[randomeFeatureSameSize]
        y_Test_Random = y_test_Random_Div

        # Train with method 2
        clfRandom = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clfRandom.fit(X_train_Random_Div, y_Train_Random_Div)
        y_Pred_Random = clfRandom.predict(X_Test_Random)
        acc_random += metrics.accuracy_score(y_Test_Random, y_Pred_Random.round())
        mcc_random += metrics.matthews_corrcoef(y_Test_Random, y_Pred_Random.round())
        auc_random += metrics.roc_auc_score(y_Test_Random, y_Pred_Random.round())

        # Train with method 1
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train_IF_Div,
                y_train_IF_Div)  # Build a forest of trees from the training set (X, y).
        y_Predict_IF = clf.predict(X_Test_IF)

        acc_if += metrics.accuracy_score(y_Test_IF, y_Predict_IF.round())
        mcc_if += metrics.matthews_corrcoef(y_Test_IF, y_Predict_IF.round())
        auc_if += metrics.roc_auc_score(y_Test_IF, y_Predict_IF.round())
    printResult(acc_random, mcc_random, auc_random, acc_if, mcc_if, auc_if, nTimes)