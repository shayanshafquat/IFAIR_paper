import csv
import sklearn
from sklearn.linear_model import LogisticRegression
from csv import reader

TrainInput = []
with open('iFairDataTrain.csv', 'r') as obj:
    csv_reader = reader(obj)
    temp = list(csv_reader)
    for row in temp :
        TrainInput.append(row)

TrainOutput = []
with open('iFairTrainDataOutput.csv', 'r') as obj:
    csv_reader = reader(obj)
    temp = list(csv_reader)
    for row in temp :
        TrainOutput.append(row)

TestInput = []
with open('iFairDataTest.csv', 'r') as obj:
    csv_reader = reader(obj)
    temp = list(csv_reader)
    for row in temp :
        TestInput.append(row)

TestOutput = []
with open('iFairTestDataOutput.csv', 'r') as obj:
    csv_reader = reader(obj)
    temp = list(csv_reader)
    for row in temp :
        TestOutput.append(row)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(TrainInput, TrainOutput)
LR.predict(TestInput)
print(round(LR.score(TestInput,TestOutput), 4))

