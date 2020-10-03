from a8_corr_svm_selftest import correction_svm_selftest
from a_utils import *
team_file = getNewDataset()
print("A8./ 70:30 - Tự Train Test - SVM to Predict")
correction_svm_selftest(team_file.train, team_file.resultColName, 1, 0.2, IF_Method.UnivariateSelection, 20)
correction_svm_selftest(team_file.train, team_file.resultColName, 1, 0.2, IF_Method.PearsonCorrelationMatrix, 20)
correction_svm_selftest(team_file.train, team_file.resultColName, 1, 0.2, IF_Method.FeatureImportance, 20)
