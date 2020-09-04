import numpy as np
import csv
import scipy
import sklearn
import math
from csv import reader
import scipy.optimize as optmz
from sklearn import preprocessing

input = []
with open('PrunedTest.csv', 'r') as obj:
    csv_reader = reader(obj)
    temp = list(csv_reader)
    for row in temp:
        input.append(row)

input = np.array(input)
output = np.array(input[:, -1]).flatten()
input = input[:, :-1]
input = preprocessing.scale(input)
sensitive = input[:, -1]
non_sensitive = input[:, :-1]

# K = [10, 20, 30]
Lambda = [0, 0.05, 0.1, 1, 10, 100]
Myu = [0, 0.05, 0.1, 1, 10, 100]

def iFair (pars, data_sensitive, data_nonsensitive, input, K, Lambda, Myu, results=0) :
    distance = []
    dis = []
    xistar = []
    txistar = []
    params = []
    tparams = []
    for i in range(0, len(pars)+1) :
        if(i != 0 and i%11 == 0) :
            params.append(tparams)
            tparams = []
            if(i != 121) :
                tparams.append(pars[i])
        else :
            tparams.append(pars[i])
    for i in input :
        dis = []
        for j in range(1,K+1) :
            temp = 0
            for k in range(0, 11) :
                temp += params[0][k]*((i[k]-params[j][k])*(i[k]-params[j][k]))
            temp = pow(temp, 0.5)
            dis.append(temp)
        distance.append(dis)
    ct = 0
    for i in range(0, len(distance)) :
        sum = 0
        prod = 0
        for j in distance[i] :
            sum += math.exp((-1)*(j))
        for j in range(1, K+1) :
            txistar = []
            txistar = params[j]
            prod = math.exp((-1)*distance[i][j-1])
            prod /= sum
            for k in range(0, len(txistar)) :
                txistar[k] *= prod
            if(j == 1) :
                xistar.append(txistar)
            else :
                for k in range(0, len(txistar)) :
                    xistar[ct][k] += txistar[k]
        ct += 1

    Lutil = 0
    Lfair = 0
    Total_Loss = 0
    for i in range(0, len(input)) :
        for j in range(0, len(input[0])) :
            Lutil += ((input[i][j] - xistar[i][j]) * (input[i][j] - xistar[i][j]))
            print(Lutil)
            print("yo")
    Lutil *= Lambda

    for i in range(0, len(input)) :
        for j in range(0, len(input)) :
            disxistar = 0
            disxsens = 0
            for k in range(0, 11) :
                disxistar += params[0][k]*((xistar[i][k] - xistar[j][k]) * (xistar[i][k] - xistar[j][k]))
            disxistar = pow(disxistar, 0.5)
            for k in range(0, 10) :
                disxsens += params[0][k]*((data_nonsensitive[i][k] - data_nonsensitive[j][k]) * (data_nonsensitive[i][k] - data_nonsensitive[j][k]))
            disxsens = pow(disxsens, 0.5)
            Lfair += ((disxistar - disxsens) * (disxistar - disxsens))
    Lfair *= Myu

    Total_Loss = Lutil + Lfair
    return Total_Loss



k = 10
tempFinalValue10 = 150000
InitialGuess = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10]]
#bound = [(None, None), (None, None), (0, 1), (None, None), (0, 1), (None, None), (0, 1), (0, 1), (0, 1), (None, None),
         #(None, None), (None, None)]
for i in Lambda:
    for j in Myu:
        FinalValue10 = optmz.fmin_l_bfgs_b(iFair, x0=InitialGuess, epsilon=1e-5,
                                           args=(sensitive, non_sensitive, input, k, i, j, 0),
                                           approx_grad=True, maxfun=150000,
                                           maxiter=150000)
        if (FinalValue10[1] < tempFinalValue10):
            tempFinalValue10 = FinalValue10[1]
            FinalGuess10 = FinalValue10[0]

k = 20
tempFinalValue20 = 150000
IntialGuess = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10]]
#bound = [(None, None), (None, None), (0, 1), (None, None), (0, 1), (None, None), (0, 1), (0, 1), (0, 1), (None, None),
         #(None, None), (None, None)]
for i in Lambda:
    for j in Myu:
        FinalValue20 = optmz.fmin_l_bfgs_b(iFair, x0=InitialGuess, epsilon=1e-5,
                                           args=(sensitive, non_sensitive, k, i, j, 0),
                                           approx_grad=True, maxfun=150000,
                                           maxiter=150000)
        if (FinalValue20[1] < tempFinalValue20):
            tempFinalValue20 = FinalValue20[1]
            FinalGuess20 = FinalValue20[0]

k = 30
tempFinalValue30 = 150000
IntialGuess = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10], [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10],
               [10, 1, 10, 1, 10, 1, 1, 1, 10, 10, 10]]
#bound = [(None, None), (None, None), (0, 1), (None, None), (0, 1), (None, None), (0, 1), (0, 1), (0, 1), (None, None),
         #(None, None), (None, None)]
for i in Lambda:
    for j in Myu:
        FinalValue30 = optmz.fmin_l_bfgs_b(iFair, x0=InitialGuess, epsilon=1e-5,
                                           args=(sensitive, non_sensitive, k, i, j, 0),
                                           approx_grad=True, maxfun=150000,
                                           maxiter=150000)
        if (FinalValue30[1] < tempFinalValue30):
            tempFinalValue30 = FinalValue30[1]
            FinalGuess30 = FinalValue30[0]

FinalValue = 0
if (tempFinalValue10 <= tempFinalValue20 and tempFinalValue10 <= tempFinalValue30):
    FinalValue = tempFinalValue10
    FinalGuess = FinalGuess10
    K = 10
elif (tempFinalValue20 <= tempFinalValue10 and tempFinalValue20 <= tempFinalValue30):
    FinalValue = tempFinalValue20
    FinalGuess = FinalGuess20
    K = 20
elif (tempFinalValue30 <= tempFinalValue10 and tempFinalValue30 <= tempFinalValue20):
    FinalValue = tempFinalValue30
    FinalGuess = FinalGuess30
    K = 30

distance = []
params = []
tparams = []
xistar = []
for i in range(0, len(FinalGuess)+1) :
    if(i != 0 and i%11 == 0) :
        params.append(tparams)
        tparams = []
        if(i != 121) :
            tparams.append(FinalGuess[i])
        else :
            tparams.append(FinalGuess[i])
for i in input:
    dis = []
    for j in range(1, K + 1):
        temp = 0
        for k in range(0, 11):
            temp += params[0][k] * ((i[k] - params[j][k]) * (i[k] - params[j][k]))
        temp = pow(temp, 0.5)
        dis.append(temp)
    distance.append(dis)
ct = 0
for i in range(0, len(distance)):
    sum = 0
    prod = 0
    for j in distance[i]:
        sum += math.exp((-1) * (j))
    for j in range(1, K + 1):
        txistar = []
        txistar = params[j]
        prod = math.exp((-1) * distance[i][j - 1])
        prod /= sum
        for k in range(0, len(txistar)):
            txistar[k] *= prod
        if (j == 1):
            xistar.append(txistar)
        else:
            for k in range(0, len(txistar)):
                xistar[ct][k] += txistar[k]
    ct += 1


with open('iFairDataTest.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(xistar)

with open('iFairTestDataOutput.csv', 'w', newline='') as csvfile :
    writer = csv.writer(csvfile)
    writer.writerows(output)

