from csp import CSP
from feature_extraction import dataloader
#import evaluation
from sklearn.naive_bayes import MultinomialNB
from backtrackingsearch import BacktrackingSearch
import math
from constraint import *
# Train Classifiers for each label type

loader = dataloader.DataLoader('data/train.csv')
trainX = loader.extractNBCountMatrix()
trainlabelindices = loader.extractLabelIndices()
#evaluator = evaluation.Evaluator()
classifiers = {}
#testx = trainX[0]
testx = trainX[0] 
for labeltype in ['sentiment', 'event', 'time']:
	nbclassifier = MultinomialNB()
	y = trainlabelindices[labeltype]
	nbclassifier.fit(trainX, y)
	classifiers[labeltype] = nbclassifier
	#print nbclassifier.predict_proba(testx)

# create a CSP with a variable for each possible label in each label class with a 
# domain of 0, 1 with a unary potential that corresponds to exp(P(y|x)) (this is the posterior 
# where the total probability is associated with just the other labels in the class- will this cause
# problems? Otherwise, I will have to train on the entire features set which would mean I would 
#either have to train nb on individual labels and split up each example into multiple examples one from
# each class or plug in the entire vector and ... 
# 
testcsp = CSP()
backsearch = BacktrackingSearch()

# add a variable and unary potential for each label 
numlabelsdict = loader.getNumLabels()
for labeltype in ['sentiment', 'time', 'event']:
	numlabels = numlabelsdict[labeltype] 
	labelprobabilities= classifiers[labeltype].predict_proba(testx)# testx needs to be in the correct format
	for index in range(numlabels):
		varname = labeltype + str(index)
		testcsp.add_variable(varname, [0,1])
		score = labelprobabilities[0][index]
		print varname, score
		testcsp.add_unary_potential(varname, lambda x: math.pow(score, x) * math.pow(1-score, 1-x))
backsearch.solve(testcsp, False, False, False)#gives memory error for full set of labels, need to switch to actual library
print backsearch.optimalAssignment

	
