from a1_corr_matrix import correction_matrix
from a7_corr_svm import correction_svm
from a5_kfold_svm import kfold_svm
from a3_corr_kfold import correction_kfold
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
    print(textcolor_display("Algorithms you want to use: ", color.WARNING))
    print("1. Correclation Matrix")
    print("2. Correclation Matrix - Self Test")
    print("3. Correclation Matrix - K-fold")
    print("5. KFold SVM")
    print("7. Correclation SVM")
    print("8. Correclation SVM - Self Test")
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
    if algo_choice == 3:
        nTimes = int(input("Time Run (1-100): "))
        percentTest = float(input("Percent Test(0-0.9): "))
        coef = float(input("Coef(0-0.9): "))
        numK = int(input("Number K-Fold (5 or 10): "))
        correction_kfold(team_file.train, team_file.resultColName,
                         team_file.listFileTest, nTimes, percentTest, coef, numK)
    if algo_choice == 5:
        nTimes = int(input("Time Run (1-100): "))
        percentTest = float(input("Percent Test(0-0.9): "))
        coef = float(input("Coef(0-0.9): "))
        numK = int(input("Number K-Fold (5 or 10): "))
        kfold_svm(team_file.train, team_file.resultColName,
                  team_file.listFileTest, nTimes, percentTest, coef, numK)
    if algo_choice == 7:
        print(textcolor_display("7. Correclation SVM", color.OKBLUE))
        nTimes = int(input("Time Run (1-100): "))
        percentTest = float(input("Percent Test(0-0.9): "))
        coef = float(input("Coef(0-0.9): "))
        correction_matrix(team_file.train, team_file.resultColName,
                          team_file.listFileTest, nTimes, percentTest, coef)
    if algo_choice == 8:
        print(textcolor_display("8. Correclation SVM - Self Test", color.OKBLUE))
        nTimes = int(input("Time Run (1-100): "))
        percentTest = float(input("Percent Test(0-0.9): "))
        coef = float(input("Coef(0-0.9): "))
        correction_matrix(team_file.train, team_file.resultColName,
                          team_file.listFileTest, nTimes, percentTest, coef)
    print(textcolor_display("Run stimulations again? ", color.WARNING))
    again_no = str(input(textcolor_display("Enter no for stop, otherwise : ", color.OKBLUE)))
    if 'no' in again_no:
        print(textcolor_display("Terminated the stimulations", color.FAIL))
        break
    else:
        print(textcolor_display("========= Welcome back =========", color.OKGREEN))
