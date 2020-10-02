import sys
from a5_kfold_svm import kfold_svm
from a_utils import *
team_file = getOldDataset()
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 0.1, 5)
