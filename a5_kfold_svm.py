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

# Revise is needed
# Train on one dataset, then test of another dataset
def kfold_svm(filenameTrain, resultColName, nTimes, coef_percent, numK, flag, nlargestFeatures):
    ways_to_if = "NONE"
    if (flag == IF_Method.PearsonCorrelationMatrix):
        ways_to_if = "Pearson Correlation Matrix"
    if (flag == IF_Method.UnivariateSelection):
        ways_to_if = "Univariate Selection"
    if (flag == IF_Method.FeatureImportance):
        ways_to_if = "Feature Importance"
    print("Cách để chọn features : " + ways_to_if)
    print(str("Train bằng file ") + str(filenameTrain))
    print("KFold = " + str(numK))
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
    numbers = rng.choice(len(colName) - 2, size=len(importanceFeature), replace=False)
    randomeFeatureSameSize = colName.delete(0).take(numbers)
    X_Train_Random = df[randomeFeatureSameSize]
    y_Train_Random = y
    X_Train_ImportFeature = df[importanceFeature]
    y_Train_ImportFeature = y
    print("KFold tự kiểm tra chéo. Chạy lặp " + str(nTimes) + " lần.")
    acc_random = 0.0
    mcc_random = 0.0
    auc_random = 0.0
    acc_if = 0.0
    mcc_if = 0.0
    auc_if = 0.0
    for n in range(nTimes):
        if nTimes == 0:
            break
        kf = KFold(n_splits=int(numK))
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
            accuracy_model_acc.append(metrics.accuracy_score(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
            accuracy_model_mcc.append(
                metrics.matthews_corrcoef(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
            accuracy_model_auc.append(metrics.roc_auc_score(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
        accuracy_model_acc_random = []
        accuracy_model_mcc_random = []
        accuracy_model_auc_random = []
        X_new_kfold = X_Train_Random
        y_new_kfold = y_Train_Random
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
                metrics.accuracy_score(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
            accuracy_model_mcc_random.append(
                metrics.matthews_corrcoef(y_test_kfold_loops, model.predict(X_test_kfold_loops)))
            accuracy_model_auc_random.append(
                metrics.roc_auc_score(y_test_kfold_loops, model.predict(X_test_kfold_loops)))

        acc_if += float(sum(map(float, accuracy_model_acc)) / len(accuracy_model_acc))
        mcc_if += float(sum(map(float, accuracy_model_mcc)) / len(accuracy_model_mcc))
        auc_if += float(sum(map(float, accuracy_model_auc)) / len(accuracy_model_auc))
        acc_random += float(sum(map(float, accuracy_model_acc_random)) / len(accuracy_model_acc_random))
        mcc_random += float(sum(map(float, accuracy_model_mcc_random)) / len(accuracy_model_mcc_random))
        auc_random += float(sum(map(float, accuracy_model_auc_random)) / len(accuracy_model_auc_random))
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
