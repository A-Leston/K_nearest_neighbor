import numpy as np

from sklearn import neighbors, datasets

n_neighbors = 1  # K in KNN classifier  (lower seems better)
iris = datasets.load_iris()

alldata = iris.data
alltarget = iris.target
print("all given data points: \n", alldata)
# print("all given targets (classifications): \n", alltarget)

trainData = []  		# will have all train data of three classes
trainTarget = []  		# will have all class labels of the train data

testData = []  			# will have all test data of three classes for prediction test
testClasses = []		# will have the corresponding classes
setSize = 34	   		# dictates size of training set, max is 50 (would be 100% of data for training but dont do that)

# Class 0 data separation
for i in range(0, setSize):
	trainData.append(alldata[i])
	trainTarget.append(alltarget[i])
for i in range(setSize, 50):
	testData.append(alldata[i])
	testClasses.append(alltarget[i])

# Class 1 data separation
for i in range(50, 50 + setSize):
	trainData.append(alldata[i])
	trainTarget.append(alltarget[i])
for i in range(50 + setSize, 100):
	testData.append(alldata[i])
	testClasses.append(alltarget[i])

# Class 2 data separation
for i in range(100, 100 + setSize):
	trainData.append(alldata[i])
	trainTarget.append(alltarget[i])
for i in range(100 + setSize, 150):
	testData.append(alldata[i])
	testClasses.append(alltarget[i])

'''
print("all the given data used for training:")
for j in range(len(trainData)):
	print(trainData[j])
'''

'''
print("correct classes for the data used in training:")
j = 0
for x in range(3):
	print(trainTarget[j:j+setSize])
	j += setSize
'''

'''
print("all the data used for prediction test:")
for z in range(len(testData)):
	print(testData[z])
'''

nn = neighbors.KNeighborsClassifier(n_neighbors)
nn.fit(trainData, trainTarget)  		# Training is done
predictions = nn.predict(testData)  	# testing

print("correct classes for prediction test set: \n", testClasses)
print("class predictions: \n", list(predictions))
print("---------stats---------")
print("knn =", n_neighbors)
print("%", setSize/50 * 100, "of the data set was used for training")
errors = 0
predictions = list(predictions)			# checking for errors
for x in range(len(testClasses)):
	if predictions[x] != testClasses[x]:
		errors += 1
print(errors, "error(s)")
