import sys
from a7_corr_svm import correction_svm
from a_utils import *
team_file = getOldDataset()
correction_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 0.1)
