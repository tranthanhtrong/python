import sys
from a3_corr_kfold import correction_kfold
from a_utils import *
team_file = getOldDataset()
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 0.1, 5)
