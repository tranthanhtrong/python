from a5_kfold_svm import kfold_svm
from a_utils import *

team_file = getOldDataset()
print("A5./ KFold, kiểm chéo, SVM to Predict")
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.3, 5, IF_Method.UnivariateSelection,
          25)
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.3, 5,
          IF_Method.PearsonCorrelationMatrix, 25)
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.3, 5, IF_Method.FeatureImportance, 25)
