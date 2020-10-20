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
from mpmath import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import seaborn as sns
import networkx as nx
from textwrap import wrap
from sklearn.feature_selection import SelectFromModel

def printGraph(columns, data):
    # plt.style.use('ggplot')
    columns = columns.delete(len(columns) - 1)
    data = data[columns]
    corData = data.corr()
    # links = corData.stack().reset_index()
    # links.columns = ['var1', 'var2', 'value']
    # links_filtered = links.loc[(links['value'] > 0.8) & (links['var1'] != links['var2'])]
    # G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    # nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corData, annot=False, cmap=plt.cm.Reds,ax=ax)
    plt.show()


class TeamFile:
    # instance attribute
    def __init__(self, train, listFileTest, resultColName):
        self.train = train
        self.listFileTest = listFileTest
        self.resultColName = resultColName


def getOldDataset():
    train = "feng_x.csv"
    fileListTest = []
    fileListTest.append('yu_x.csv')
    fileListTest.append('zeller_x.csv')
    fileListTest.append('vogtmann_x.csv')
    return TeamFile(train, fileListTest, "RS")


def getNewDataset():
    train = "ibdfullHS_UCr_x.csv" #iCDr & UCf &iCDf &CDr&CDf
    fileListTest = []
    fileListTest.append('ibdfullHS_iCDr_x.csv')
    fileListTest.append('ibdfullHS_UCf_x.csv')
    fileListTest.append('ibdfullHS_iCDf_x.csv')
    fileListTest.append('ibdfullHS_CDr_x.csv')
    fileListTest.append('ibdfullHS_CDf_x.csv')
    return TeamFile(train, fileListTest, "RS")


class color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def textcolor_display(text, values):
    return f"{values}" + text + f"{color.ENDC}"


class IF_Method:
    PearsonCorrelationMatrix = 1
    UnivariateSelection = 2
    FeatureImportance = 3


def findImportancesFeatures(resultColName, filenameTrain, coef_percent, flag, nlargestFeatures):
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
        print("Number feature selected : " + str(len(importanceFeature)))
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
    X_Train_ImportFeature = df[importanceFeature]
    y_Train_ImportFeature = y
    printGraph(importanceFeature, data)
    return importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature

def findRandomeFeaturesSets(resultColName, filenameTrain, sizeIF):
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    y = df[resultColName]
    rng = default_rng()
    # In colName has n columns, position of RS is n - 1. Because of a noname rows of V1,V2,V3,...
    numbers = rng.choice(len(colName) - 2, size=sizeIF, replace=False)
    randomeFeatureSameSize = colName.delete(0).take(numbers)
    X_Train_Random = df[randomeFeatureSameSize]
    y_Train_Random = y
    return randomeFeatureSameSize, X_Train_Random, y_Train_Random


def printResult(acc_random, mcc_random, auc_random, acc_if, mcc_if, auc_if, nTimes):
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
def sumThenAveragePercisely(accuracy_model_acc):
    return fdiv(fsum(accuracy_model_acc), len(accuracy_model_acc), prec=5)
