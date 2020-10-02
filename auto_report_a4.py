import sys
from a4_corr_kfold_selftest import correction_kfold_selftest
from a_utils import *
team_file = getOldDataset()
correction_kfold_selftest(team_file.train, team_file.resultColName, 1, 0.1, 0.1, 5)