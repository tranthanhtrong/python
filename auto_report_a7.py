import sys
from a7_corr_svm import correction_svm
from a_utils import *
team_file = getOldDataset()
print("A7./ 70:30 Chéo, SVM to Predict")
correction_svm(team_file.train, team_file.resultColName,
                  team_file.listFileTest, 1, 0.1, IF_Method.UnivariateSelection, 10)
correction_svm(team_file.train, team_file.resultColName,
                  team_file.listFileTest, 1, 0.1, IF_Method.PearsonCorrelationMatrix, 10)
correction_svm(team_file.train, team_file.resultColName,
                  team_file.listFileTest, 1, 0.1, IF_Method.FeatureImportance, 10)
