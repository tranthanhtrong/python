from a1_corr_matrix import correction_matrix
from a5_kfold_svm import kfold_svm
from a_utils import *
while True:
  print(getMainTextColor("========= Stimulates Start ========="))
  #Choose datasets
  print(getMainTextColor("Dataset you want to run on: "))
  print(getMainTextColor("1. Old datasets"))
  print(getMainTextColor("2. New datasets"))
  data_set_choice = int(input("Input : "))
  if data_set_choice == 1:
	  team_file = getOldDataset()
  else:
    team_file = getNewDataset()
  #Choose Algorithms
  print(getMainTextColor("Algorithms you want to use: "))
  print(getSubTextColor("1. Correclation Matrix"))
  print(getSubTextColor("5. KFold SVM"))
  algo_choice = int(input("Input : "))
  if algo_choice == 1:
    nTimes = int(input(getInputColor("Time Run (1-100): ")))
    percentTest = float(input(getInputColor("Percent Test(0-0.9): ")))
    coef = float(input(getInputColor("Coef(0-0.9): ")))
    correction_matrix(team_file.train, team_file.resultColName,
		                  team_file.listFileTest, nTimes, percentTest, coef)
  if algo_choice == 5:
    nTimes = int(input(getInputColor("Time Run (1-100): ")))
    percentTest = float(input(getInputColor("Percent Test(0-0.9): ")))
    coef = float(input(getInputColor("Coef(0-0.9): ")))
    numK = int(input(getInputColor("Number K-Fold (5 or 10): ")))
    kfold_svm(team_file.train, team_file.resultColName,
		          team_file.listFileTest, nTimes, percentTest, coef, numK)
  print(getMainTextColor("Run stimulations again? "))
  again_no = str(input("Enter no for stop, otherwise : "))
  if 'no' in again_no:
    print(getTerminatedColor("Terminated the stimulations"))
    break
  else:
    print(getMainTextColor("========= Welcome back ========="))
