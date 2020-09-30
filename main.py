from a1_corr_matrix import correction_matrix
from a5_kfold_svm import kfold_svm
from a2_corr_selftest import correction_matrix_selftest
from a_utils import *
while True:
    print("========= Stimulates Start =========")
    # Choose datasets
    print("Dataset you want to run on: ")
    print("1. Old datasets")
    print("2. New datasets")
    data_set_choice = int(input("Input : "))
    if data_set_choice == 1:
        team_file = getOldDataset()
    else:
        team_file = getNewDataset()
    # Choose Algorithms
    print("Algorithms you want to use: ")
    print("1. Correclation Matrix")
    print("2. Correclation Matrix - Self Test")
    print("5. KFold SVM")
    algo_choice = int(input("Input : "))
    if algo_choice == 1:
        nTimes = int(input("Time Run (1-100): "))
        percentTest = float(input("Percent Test(0-0.9): "))
        coef = float(input("Coef(0-0.9): "))
        correction_matrix(team_file.train, team_file.resultColName,
                          team_file.listFileTest, nTimes, percentTest, coef)
    if algo_choice == 2:
        nTimes = int(input("Time Run (1-100): "))
        percentTest = float(input("Percent Test(0-0.9): "))
        coef = float(input("Coef(0-0.9): "))
        correction_matrix_selftest(team_file.train, team_file.resultColName,
                                   nTimes, percentTest, coef)
    if algo_choice == 5:
        nTimes = int(input("Time Run (1-100): "))
        percentTest = float(input("Percent Test(0-0.9): "))
        coef = float(input("Coef(0-0.9): "))
        numK = int(input("Number K-Fold (5 or 10): "))
        kfold_svm(team_file.train, team_file.resultColName,
                  team_file.listFileTest, nTimes, percentTest, coef, numK)
    print("Run stimulations again? ")
    again_no = str(input("Enter no for stop, otherwise : "))
    if 'no' in again_no:
        print("Terminated the stimulations")
        break
    else:
        print("========= Welcome back =========")
        print("========= Welcome back =========")
