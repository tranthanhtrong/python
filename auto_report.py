import sys
from a1_corr_matrix import correction_matrix
from a_utils import *
team_file = getOldDataset()
correction_matrix(team_file.train, team_file.resultColName,
                      team_file.listFileTest, 10, 0.1, 0.1)
