class TeamFile:
    # instance attribute
    def __init__(self, train, listFileTest, resultColName):
        self.train = train
        self.listFileTest = listFileTest
        self.resultColName = resultColName


def getOldDataset():
    train = "feng_x.csv"
    fileListTest = []
    # fileListTest.append('yu_x.csv')
    fileListTest.append('zeller_x.csv')
    fileListTest.append('vogtmann_x.csv')
    return TeamFile(train, fileListTest, "RS")


def getNewDataset():
    train = "ibdfullHS_UCr_x.csv"
    fileListTest = []
    fileListTest.append('ibdfullHS_CDr_x.csv')
    fileListTest.append('ibdfullHS_iCDf_x.csv')
    fileListTest.append('ibdfullHS_iCDr_x.csv')
    fileListTest.append('ibdfullHS_UCf_x.csv')
    return TeamFile(train, fileListTest, "RS")


class IF_Method:
    PearsonCorrelationMatrix = 1
    UnivariateSelection = 2
    FeatureImportance = 3


def textcolor_display(text, values):
    return text
