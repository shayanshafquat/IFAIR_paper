import numpy as np
import csv
import sklearn
from sklearn.linear_model import LogisticRegression
from csv import reader
from sklearn import preprocessing

TrainInput = []
with open('train_german.csv', 'r') as obj:
    csv_reader = reader(obj)
    temp = list(csv_reader)
    for row in temp :
        TrainInput.append(row)

TrainInput = np.array(TrainInput)
TrainOutput = np.array(TrainInput[:,-1]).flatten()
TrainInput = TrainInput[:,:-1]
TrainInput = preprocessing.scale(TrainInput)

TestInput = []
with open('test_german.csv', 'r') as obj:
    csv_reader = reader(obj)
    temp = list(csv_reader)
    for row in temp :
        TestInput.append(row)

TestInput = np.array(TestInput)
TestOutput = np.array(TestInput[:,-1]).flatten()
TestInput = TestInput[:,:-1]
TestInput = preprocessing.scale(TestInput)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(TrainInput, TrainOutput)
LR.predict(TestInput)
print(round(LR.score(TestInput,TestOutput), 4))

