from a1_corr_matrix import correction_matrix
from a5_kfold_svm import kfold_svm
from a_utils import *
import enquiries
while True:
  print(getMainTextColor("========= Stimulates Start ========="))
  datasets = ['1. Old Datasets', '2. New Datasets']
  algo = ['1. Correclation Matrix', '5. KFold SVM']
  runTimes = ['1','2','3','4','5']
  pecentages = ['0.1','0.2', '0.3', '0.4', '0.5', '0.6', '0.7']
  k_fold = ['5', '10']
  termis = ['Stop', 'Continue']
  choices = enquiries.choose("Dataset you want to run on: ", datasets)
  #Choose datasets
  if '1' in choices:
	  team_file = getOldDataset()
  else:
    team_file = getNewDataset()
  #Choose Algorithms
  choiceAlgo = enquiries.choose("Algorithms you want to use: ", algo)
  if '1' in choiceAlgo:
    nTimes = int(enquiries.choose("How many times you want to run? ", runTimes))
    percentTest = float(enquiries.choose("How much use for test? ", pecentages))
    coef = float(enquiries.choose("How much coef? ", pecentages))
    correction_matrix(team_file.train, team_file.resultColName,
		                  team_file.listFileTest, nTimes, percentTest, coef)
  if '5' in choiceAlgo:
    nTimes = int(enquiries.choose("How many times you want to run? ", runTimes))
    percentTest = float(enquiries.choose("How much use for test? ", pecentages))
    coef = float(enquiries.choose("How much coef? ", pecentages))
    numK = int(enquiries.choose("How many K-Fold? ", k_fold))
    kfold_svm(team_file.train, team_file.resultColName,
		          team_file.listFileTest, nTimes, percentTest, coef, numK)
  getTerm = enquiries.choose("Run stimulations again? ", termis)
  if 'Stop' in getTerm:
    print(getTerminatedColor("Terminated the stimulations"))
    break
  else:
    print(getMainTextColor("========= Welcome back ========="))
