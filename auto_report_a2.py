import sys
from a2_corr_selftest import correction_matrix_selftest
from a_utils import *
team_file = getOldDataset()
correction_matrix_selftest(team_file.train, team_file.resultColName, 1, 0.1, 0.1)
