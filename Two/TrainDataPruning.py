import csv

with open('adult.data.txt') as input_file:
   lines = input_file.readlines()
   newLines = []
   for line in lines:
      newLine = line.strip().split(', ')
      newLines.append(newLine)

siz = len(newLines)

ct1 = 0
ct2 = 0
tempSex = 0
tempSalary = 0
tempData = []
finalData = []
for i in newLines[:siz-1] :
    ct2=0
    tempData.clear()
    for j in i :
        if(ct2 == 0 or ct2 == 2 or ct2 == 4 or ct2 == 10 or ct2 == 11 or ct2 == 12) :
            tempData.append(int(newLines[ct1][ct2],10))
        elif(ct2 == 1) :
            if(j == "Private" or j == "Self-emp-not-inc" or j == "Self-amp-inc" or j == "Federal-gov" or j == "Local-gov" or j == "State-gov") :
                tempData.append(1)
            else :
                tempData.append(0)
        elif(ct2 == 3) :
            if(j == "Bachelors" or j == "HS-grad" or j == "Prof-school" or j == "Assoc-adm" or j == "Assoc-voc" or j == "Masters" or j == "Doctorate") :
                tempData.append(1)
            else :
                tempData.append(0)
        elif(ct2 == 5) :
            if(j == "Married-civ-spouse" or j == "Married-AF-spouse" or j == "Married-spouse-absent") :
                tempData.append(1)
            else :
                tempData.append(0)
        elif(ct2 == 6) :
            if(j == "Tech-support" or j == "Sales" or j == "Exec-managerial" or j == "Prof-speciality" or j == "Machine-op-inspct" or j == "Adm-clerical" or j == "Armed-Forces") :
                tempData.append(1)
            else :
                tempData.append(0)
        elif(ct2 == 9) :
            if(j == "Male") :
                tempSex = 1
            else :
                tempSex = 0
        elif(ct2 == 14) :
            if(j == ">50K") :
                tempSalary = 1
            else :
                tempSalary = 0
        ct2+=1
    tempData.append(tempSex)
    tempData.append(tempSalary)
    finalData.append(tempData[:])
    ct1+=1

with open('PrunedTrain.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(finalData)