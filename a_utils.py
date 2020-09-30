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


import os, sys


def has_colors():
    if ((os.getenv("CLICOLOR", "1") != "0" and sys.stdout.isatty()) or
            os.getenv("CLICOLOR_FORCE", "0") != "0"):
        return True
    else:
        return False


def textcolor_display(text, type_mes='er'):
    if has_colors():
        end = '\x1b[0m'
        if type_mes in ['er', 'error']:
            begin = '\x1b[1;33;41m'
            return begin + text + end

        if type_mes in ['inf', 'information']:
            begin = '\x1b[0;36;44m'
            return begin + text + end
    else:
        return text
