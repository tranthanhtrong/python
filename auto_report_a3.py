from a3_corr_kfold import correction_kfold
from a_utils import *

team_file = getNewDataset()
print("A3./ KFold, Kiểm tra Chéo, RandomForestClassifier to Predict")
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 5,
                 IF_Method.UnivariateSelection, 10)
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 5,
                 IF_Method.PearsonCorrelationMatrix, 10)
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.1, 5,
                 IF_Method.FeatureImportance, 10)
