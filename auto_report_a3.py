from a3_corr_kfold import correction_kfold
from a_utils import *

team_file = getOldDataset()
print("A3./ KFold, Kiểm tra Chéo, RandomForestClassifier to Predict")
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.2, 5,
                 IF_Method.UnivariateSelection, 20)
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.2, 5,
                 IF_Method.PearsonCorrelationMatrix, 20)
correction_kfold(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.2, 5,
                 IF_Method.FeatureImportance, 20)
