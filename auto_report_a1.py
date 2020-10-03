from a1_corr_matrix import correction_matrix
from a_utils import *

team_file = getNewDataset()
print("A1./ 70:30 Chéo, RandomForestClassifier to Predict")
correction_matrix(team_file.train, team_file.resultColName,
                  team_file.listFileTest, 1, 0.2, IF_Method.UnivariateSelection, 20)
correction_matrix(team_file.train, team_file.resultColName,
                  team_file.listFileTest, 1, 0.2, IF_Method.PearsonCorrelationMatrix, 20)
correction_matrix(team_file.train, team_file.resultColName,
                  team_file.listFileTest, 1, 0.2, IF_Method.FeatureImportance, 20)
