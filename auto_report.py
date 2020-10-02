import sys
from a1_corr_matrix import correction_matrix
from a7_corr_svm import correction_svm
from a5_kfold_svm import kfold_svm
from a6_kfold_svm_selftest import kfold_svm_sefltest
from a3_corr_kfold import correction_kfold
from a4_corr_kfold_selftest import correction_kfold_selftest
from a2_corr_selftest import correction_matrix_selftest
from a8_corr_svm_selftest import correction_svm_selftest
from a_utils import *
team_file = getOldDataset()
correction_matrix(team_file.train, team_file.resultColName,
                      team_file.listFileTest, 1, 0.1, 0.1)
correction_matrix_selftest(team_file.train, team_file.resultColName, 1, 0.1, 0.1)
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 0.1, 5) 
correction_kfold_selftest(team_file.train, team_file.resultColName, 1, 0.1, 0.1, 5) 
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 0.1, 5)
kfold_svm_sefltest(team_file.train, team_file.resultColName, team_file.listFileTest, 5, 0.1, 0.1, 5) 
correction_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 0.1)
correction_svm_selftest(team_file.train, team_file.resultColName, 1, 0.1, 0.1)
