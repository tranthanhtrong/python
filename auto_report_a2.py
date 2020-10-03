from a2_corr_selftest import correction_matrix_selftest
from a_utils import *

team_file = getNewDataset()
print("A2./ 70:30 - Tự Train Test - RandomForestClassifier to Predict")
correction_matrix_selftest(team_file.train, team_file.resultColName, 1, 0.2, IF_Method.UnivariateSelection, 20)
correction_matrix_selftest(team_file.train, team_file.resultColName, 1, 0.2, IF_Method.PearsonCorrelationMatrix, 20)
correction_matrix_selftest(team_file.train, team_file.resultColName, 1, 0.2, IF_Method.FeatureImportance, 20)
