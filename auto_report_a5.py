from a5_kfold_svm import kfold_svm
from a_utils import *

team_file = getNewDataset()
print("A5./ KFold, kiểm chéo, SVM to Predict")
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.2, 5, IF_Method.UnivariateSelection,
          20)
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.2, 5,
          IF_Method.PearsonCorrelationMatrix, 20)
kfold_svm(team_file.train, team_file.resultColName, team_file.listFileTest, 1, 0.2, 5, IF_Method.FeatureImportance, 20)
