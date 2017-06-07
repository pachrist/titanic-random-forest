#titanic_randforest.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv

def gender(s): #vectorizing gender (I know sklearn has functions for this but I know exactly what this does which is safer for now)
    if s == 'male': return 0
    elif s == 'female': return 1
    else: 
        print "there are empty genders" #apparently no empty genders so not a problem here
        return .5                       #if there were this could be a decent lazy mean/median imputation though

#reading the train/test csv files and getting the data into nested lists
with open('C:\\Users\\Paul\\Downloads\\train.csv','r') as csvfile:
    titanicReader = csv.reader(csvfile, delimiter=',')
    simpleData = []
    data = []
    titanicReader.next()
    for row in titanicReader:
        new_row = []
        for item in row:
            if item.isdigit() == True:
                item = float(item)  
            new_row.append(item)
        data.append(new_row)
    #observations = [float(row[5]) for row in data if row[5] != '']
    #dummy_age = median(observations) 
    for row in data:
        #if row[5] == '': row[5] = dummy_age
        if row[5] != '': row[5] = float(row[5])
        new_row = row[1:3] + [gender(row[4]),row[5]] #float(row[5])
        simpleData.append(new_row)

with open('C:\\Users\\Paul\\Downloads\\test.csv','r') as csvfile:
    testReader = csv.reader(csvfile, delimiter=',') 
    test = []
    simpleTest = []
    testReader.next()
    for row in testReader:
        new_row = []
        for item in row:
            if item.isdigit() == True:
                item = float(item) 
            new_row.append(item)
        test.append(new_row)
    #observations = [float(row[4]) for row in test if row[4] != '']
    #dummy_age = median(observations) 
    for row in test:
        #if row[4] == '': row[4] = dummy_age
        if row[4] != '': row[4] = float(row[4])
        new_row = [row[1],gender(row[3]),row[4]] #float(row[4])
        simpleTest.append(new_row)
                
#moving data into numpy matrices
features = np.matrix([row[1:4] for row in simpleData])
supervision = np.transpose(np.matrix([row[0] for row in simpleData]).ravel()) #do I need tanspose for 1D array or just for targets?
simpleTest = np.matrix(simpleTest)

classifier = RandomForestClassifier(n_estimators=100,n_jobs=4)
classifier.fit(features, supervision)
predictions = classifier.predict(test[features])

with open('C:\\Python\\titanic_cga.csv', 'wb') as csvfile:
    titanicWriter = csv.writer(csvfile, delimiter=',')
    titanicWriter.writerow(['PassengerId','Survived'])
    for i,boolean in enumerate(predictions):
        titanicWriter.writerow([i+892,int(boolean)])
