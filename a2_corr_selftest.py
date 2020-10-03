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

#Revise is needed
# Train on one dataset, then test of another dataset
def correction_matrix_selftest(filenameTrain, resultColName, nTimes, coef_percent,flag,nlargestFeatures):
    ways_to_if = "NONE"
    if (flag == IF_Method.PearsonCorrelationMatrix):
        ways_to_if = "Pearson Correlation Matrix"
    if (flag == IF_Method.UnivariateSelection):
        ways_to_if = "Univariate Selection"
    if (flag == IF_Method.FeatureImportance):
        ways_to_if = "Feature Importance"
    print("Cách để chọn features : " + ways_to_if)
    print(str("Train bằng file ") + str(filenameTrain))
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    X = df[colName]
    y = df[resultColName]
    if (flag == IF_Method.PearsonCorrelationMatrix):
        print("Hệ số tương quan > " + str(coef_percent))
        if float(coef_percent) != 0.0:
            cor = X.corr()
            cor_target = abs(cor[resultColName])
            relevant_features = cor_target[cor_target > float(coef_percent)]
        else:
            relevant_features = X_train_new
        importanceFeature = relevant_features.index
    if (flag == IF_Method.UnivariateSelection):
        print("Số lượng Importance Feature:  " + str(nlargestFeatures))
        X_No_V = X.drop(data.columns[0], 1)  # independent columns
        # y = data.iloc[:, -1]  # target column i.e price range
        # .nlargest(10, 'Score')
        # apply SelectKBest class to extract top 10 best features
        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X_No_V, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        # concat two dataframes for better visualization
        relevant_features = pd.concat([dfcolumns, dfscores], axis=1)
        relevant_features.columns = ['Specs', 'Score']
        importanceFeature = relevant_features.nlargest(nlargestFeatures, 'Score')
        importanceFeature = importanceFeature.drop('Score', 1)
        importanceFeature = importanceFeature.iloc[:, -1]
    if (flag == IF_Method.FeatureImportance):
        print("Số lượng Importance Feature:  " + str(nlargestFeatures))
        X_No_V = X.drop(data.columns[0], 1)  # independent columns
        from sklearn.ensemble import ExtraTreesClassifier
        import matplotlib.pyplot as plt
        model = ExtraTreesClassifier()
        model.fit(X_No_V, y)
        # plot graph of feature importances for better visualization
        importanceFeature = pd.Series(model.feature_importances_, index=X_No_V.columns)
        importanceFeature = importanceFeature.nlargest(nlargestFeatures)
        importanceFeature = importanceFeature.index
    rng = default_rng()
    # In colName has n columns, position of RS is n - 1. Because of a noname rows of V1,V2,V3,...
    numbers = rng.choice(len(colName) - 2, size=len(importanceFeature) - 1, replace=False)
    randomeFeatureSameSize = colName.delete(0).take(numbers)
    X_Train_Random = df[randomeFeatureSameSize]
    y_Train_Random = y
    X_Train_ImportFeature = df[importanceFeature]
    y_Train_ImportFeature = y
    X_train_IF_Div, X_test_IF_Div, y_train_IF_Div, y_test_IF_Div = train_test_split(X_Train_ImportFeature,
                                                                                    y_Train_ImportFeature,
                                                                                    test_size=0.3)
    X_train_Random_Div, X_test_Random_Div, y_Train_Random_Div, y_test_Random_Div = train_test_split(X_Train_Random,
                                                                                                    y_Train_Random,
                                                                                                    test_size=0.3)
    print("Tự kiểm tra trên 0.3 phần trăm bệnh nhân còn lại " + str(nTimes) + " lần.")
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
        clfRandom = RandomForestClassifier(n_estimators=1000, max_features='auto')
        clfRandom.fit(X_train_Random_Div, y_Train_Random_Div)
        y_Pred_Random = clfRandom.predict(X_Test_Random)
        acc_random += metrics.accuracy_score(y_Test_Random, y_Pred_Random.round())
        mcc_random += metrics.matthews_corrcoef(y_Test_Random, y_Pred_Random.round())
        auc_random += metrics.roc_auc_score(y_Test_Random, y_Pred_Random.round())

        # Train with method 1
        clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
        clf.fit(X_train_IF_Div,
                y_train_IF_Div)  # Build a forest of trees from the training set (X, y).
        y_Predict_IF = clf.predict(X_Test_IF)

        acc_if += metrics.accuracy_score(y_Test_IF, y_Predict_IF.round())
        mcc_if += metrics.matthews_corrcoef(y_Test_IF, y_Predict_IF.round())
        auc_if += metrics.roc_auc_score(y_Test_IF, y_Predict_IF.round())
    print("Khi Random ")
    print("ACC = " + str(acc_random / nTimes))
    print("MCC = " + str(mcc_random / nTimes))
    print("AUC = " + str(auc_random / nTimes))
    print("+++++ ")
    print("Khi xét Importance Features")
    print("ACC = " + str(acc_if / nTimes))
    print("MCC = " + str(mcc_if / nTimes))
    print("AUC = " + str(auc_if / nTimes))
    print("--------------------------------- ")
