from a6_kfold_svm_selftest import kfold_svm_sefltest
from a_utils import *

team_file = getNewDataset()
print("A6./ KFold tự kiểm, SVM to Predict")
kfold_svm_sefltest(team_file.train, team_file.resultColName, 1, 0.1, 5,
                   IF_Method.UnivariateSelection, 10)
kfold_svm_sefltest(team_file.train, team_file.resultColName, 1, 0.1, 5,
                   IF_Method.PearsonCorrelationMatrix, 10)
kfold_svm_sefltest(team_file.train, team_file.resultColName, 1, 0.1, 5,
                   IF_Method.FeatureImportance, 10)
