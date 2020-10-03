import sys
from a4_corr_kfold_selftest import correction_kfold_selftest
from a_utils import *

team_file = getOldDataset()
print("A4./ KFold tự kiểm, RandomForestClassifier to Predict")
correction_kfold_selftest(team_file.train, team_file.resultColName, 1, 0.1, 5, IF_Method.UnivariateSelection, 10)
correction_kfold_selftest(team_file.train, team_file.resultColName, 1, 0.1, 5, IF_Method.PearsonCorrelationMatrix, 10)
correction_kfold_selftest(team_file.train, team_file.resultColName, 1, 0.1, 5, IF_Method.FeatureImportance, 10)
