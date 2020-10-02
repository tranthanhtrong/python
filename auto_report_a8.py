import sys
from a8_corr_svm_selftest import correction_svm_selftest
from a_utils import *
team_file = getOldDataset()
correction_svm_selftest(team_file.train, team_file.resultColName, 1, 0.1, 0.1)
