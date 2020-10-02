import sys
from a6_kfold_svm_selftest import kfold_svm_sefltest
from a_utils import *
team_file = getOldDataset()
kfold_svm_sefltest(team_file.train, team_file.resultColName, team_file.listFileTest, 5, 0.1, 0.1, 5)
