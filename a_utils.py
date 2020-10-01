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
    train = "ibdfullHS_UCr_x.csv"
    fileListTest = []
    fileListTest.append('ibdfullHS_CDr_x.csv')
    fileListTest.append('ibdfullHS_iCDf_x.csv')
    fileListTest.append('ibdfullHS_iCDr_x.csv')
    fileListTest.append('ibdfullHS_UCf_x.csv')
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
def textcolor_display(text,values):
   return  f"{values}"+text+f"{color.ENDC}"
