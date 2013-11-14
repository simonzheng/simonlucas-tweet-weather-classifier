import numpy as np
from sklearn.naive_bayes import MultinomialNB
import evaluation
from feature_extraction import dataloader
import ast

loader = dataloader.DataLoader('data/train.csv')
trainX = loader.extractNBCountMatrix()
trainlabelvectors = loader.extractLabelBitStrings(.5)
evaluator = evaluation.Evaluator()
classifiers = {}
for labeltype in ['sentiment', 'event', 'time']:
	nbclassifier = MultinomialNB()
	y = trainlabelvectors[labeltype]
	nbclassifier.fit(trainX, y)
	classifiers[labeltype] = nbclassifier

# Sanity check 
#index = 9
#testY = {'sentiment':trainlabelvectors['sentiment'][index], 'event':trainlabelvectors['event'][index], 'time':trainlabelvectors['time'][index]}
#testX = trainX[0:100]
testY = trainlabelvectors
for labeltype in ['sentiment', 'event', 'time']:
	predictedvectors = []
	for example in testX:
		predicted = classifiers[labeltype].predict(testX)[0]
		predictedvectors.append(predicted)
	#expectedvectors = testY[labeltype][0:100]
	#error_rate = evaluator.error_rate(predictedvectors, expectedvectors)
	#accuracy = classifiers[labeltype].score(testX, expectedvectors)
	#print 'label: ', labeltype
	#print 'error rate: ', error_rate
	#print 'accuracy: ', accuracy