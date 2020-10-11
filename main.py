from a1_corr_matrix import correction_matrix
from a7_corr_svm import correction_svm
from a5_kfold_svm import kfold_svm
from a4_corr_kfold_selftest import correction_kfold_selftest
from a6_kfold_svm_selftest import kfold_svm_sefltest
from a3_corr_kfold import correction_kfold
from a2_corr_selftest import correction_matrix_selftest
from a_utils import *
from a8_corr_svm_selftest import correction_svm_selftest
from pathlib import Path

while True:
    print("========= Stimulates Start =========")
    print(textcolor_display("Algorithms you want to use: ", color.WARNING))
    print("1. Correclation Matrix")
    print("2. Correclation Matrix - Self Test")
    print("3. Correclation Matrix - K-fold")
    print("4. Correclation Matrix - K-fold - Selftest")
    print("5. KFold SVM")
    print("6. KFold SVM - Self Test")
    print("7. Correclation SVM")
    print("8. Correclation SVM - Self Test")
    print("9. Run all aboves!")
    print("10. Run only dataset summary!")
    algo_choice = input("Input : ")
    algo_choice = int(algo_choice)
    if algo_choice == 1:
        exec(open("auto_report_a1.py").read())
    if algo_choice == 2:
        exec(Path("auto_report_a2.py").read_text())
    if algo_choice == 3:
        exec(Path("auto_report_a3.py").read_text())
    if algo_choice == 4:
        exec(Path("auto_report_a4.py").read_text())
    if algo_choice == 5:
        exec(Path("auto_report_a5.py").read_text())
    if algo_choice == 6:
        exec(Path("auto_report_a6.py").read_text())
    if algo_choice == 7:
        exec(Path("auto_report_a7.py").read_text())
    if algo_choice == 8:
        exec(Path("auto_report_a8.py").read_text())
    if algo_choice == 9:
        exec(Path("auto_report_a1.py").read_text())
        exec(Path("auto_report_a2.py").read_text())
        exec(Path("auto_report_a3.py").read_text())
        exec(Path("auto_report_a4.py").read_text())
        exec(Path("auto_report_a5.py").read_text())
        exec(Path("auto_report_a6.py").read_text())
        exec(Path("auto_report_a7.py").read_text())
        exec(Path("auto_report_a8.py").read_text())
    if algo_choice == 10:
        exec(Path("datasets_summary.py").read_text())
    print(textcolor_display("Run stimulations again? ", color.WARNING))
    again_no = str(input(textcolor_display("Enter no for stop, otherwise : ", color.OKBLUE)))
    if 'no' in again_no:
        print(textcolor_display("Terminated the stimulations", color.FAIL))
        break
    else:
        print(textcolor_display("========= Welcome back =========", color.OKGREEN))
