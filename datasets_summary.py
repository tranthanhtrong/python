
team_file = getNewDataset()
print("Train Set Name: "+ team_file.train)
data = pd.read_csv(team_file.train)
colName = data.columns
colName = colName.delete(len(colName)-1)
print("Num Features : " + str(len(colName)))
df = pd.DataFrame(data, columns=colName)
y = data[team_file.resultColName].to_frame()
print("1 means IBD, and 0 mean Healthy Subject")
print(y[team_file.resultColName].value_counts())
print("Num Patients : " + str(len(y)))
for i in team_file.listFileTest:
    print("Test Set : " + i)
    data = pd.read_csv(i)
    colName = data.columns
    colName = colName.delete(len(colName) - 1)
    print("Num Features : " + str(len(colName)))
    df = pd.DataFrame(data, columns=colName)
    y = data[team_file.resultColName].to_frame()
    print("1 means IBD, and 0 mean Healthy Subject")
    print(y[team_file.resultColName].value_counts())
    print("Num Patients : " + str(len(y)))