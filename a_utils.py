def coloredText(r, g, b, text):
  return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def getMainTextColor(text):
  return coloredText(51,255,51,text)
def getSubTextColor(text):
  return coloredText(35,219,0,text)
def getInfoTextColor(text):
  return coloredText(255,255,0,text)
def getInputColor(text):
  return coloredText(34,139,34,text)
def getTerminatedColor(text):
  return coloredText(255,99,71,text)
def getFilenameColor(text):
  return coloredText(255,165,0,text)

class TeamFile:
    # instance attribute
  def __init__(self, train, listFileTest,resultColName):
    self.train = train
    self.listFileTest = listFileTest
    self.resultColName = resultColName

def getOldDataset():
  train = "feng_x.csv"
  fileListTest=[]
  fileListTest.append('yu_x.csv')
  fileListTest.append('zeller_x.csv')
  fileListTest.append('vogtmann_x.csv')
  return TeamFile(train,fileListTest,"RS")
def getNewDataset():
  train = "ibdfullHS_UCr_x.csv"
  fileListTest=[]
  fileListTest.append('ibdfullHS_CDr_x.csv')
  fileListTest.append('ibdfullHS_iCDf_x.csv')
  fileListTest.append('ibdfullHS_iCDr_x.csv')
  fileListTest.append('ibdfullHS_UCf_x.csv')
  return TeamFile(train,fileListTest,"RS")