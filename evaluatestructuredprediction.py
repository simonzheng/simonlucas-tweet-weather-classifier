from csp import CSP
from feature_extraction import dataloader
import evaluation
from sklearn.naive_bayes import MultinomialNB
from backtrackingsearch import BacktrackingSearch
import math
from constraint import *
from collections import Counter
import csv

#Helper method for adding binary constrains
def add_binary_constraints(constraints, csp):
	for constraint in binaryConstraints:
 		label1 = constraint[0]
 		label2 = constraint[1]
 		# We add a constraint for each pair of labels in our constraint specifying that we cannot have both of them co-occurring (i.e. turned to 1)
 		if label1 in csp.varNames and label2 in csp.varNames:
 			csp.add_binary_potential(label1, label2, lambda l1, l2 : l1 + l2 <= 1)

def loadConstraints(filename):
    with open(filename, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines())]
    return data

binaryConstraints = loadConstraints('binary-constraints.csv')

loader = dataloader.DataLoader('data/train.csv')
testloader = dataloader.DataLoader('data/test_100.csv')
#sparse matrix representing training wordcounts for MultinomialNB
trainX = loader.extractNBCountMatrix()
#sparse matrix representing test word counts for MultinomialNB
testX = loader.extractNBTestCountMatrix(testloader.corpus)

#NOTE: both label names and label indicies used
# - label indices are easier to work with for MultinomialNB class 
# because the output probabablities are order arithmetically
# - label names are easier to use for debugging purposes

#dictionary { sentiment:[] event:[] time:[] } of label names for each training example
trainlabelnames = loader.extractLabelNames()
#dictionary { sentiment:[] event:[] time:[] } of label indices for each training example
trainlabelindices = loader.extractLabelIndices()

#training confidence vector used for debugging 
trainconfidences = loader.extractLabelConfidences()


#dictionary { sentiment:[] event:[] time:[] } of label names for each training example
testlabelnames = testloader.extractLabelNames()

#dictionary { sentiment:[] event:[] time:[] } of label indices for each training example
testlabelindices = testloader.extractLabelIndices()

classifiers = {}
print "training naive bayes classifiers on test data"
# Train a multinomial naive bayes on each label type 
for labeltype in ['sentiment', 'event', 'time']:
	nbclassifier = MultinomialNB()
	# the trainY is a single index for the maximum confidence label in a label class
	y = trainlabelindices[labeltype]
	# list of all possible labels for nbclassifier
	indices = [ i for i in range(len(loader.labelnames[labeltype]))]
	# partial fit works when you don't use the full training set 
	nbclassifier.partial_fit(trainX, y, indices)
	classifiers[labeltype] = nbclassifier

print 'running csp on each example'
backsearch = BacktrackingSearch() 
#controls the minimum probability for a label to be considered in the csp
probabilitythreshold = .2
# controls the minimum confidence for a label to be present in the gold bit vector
confidencethreshold = .5
# gold output for evaluation for each training example
testgoldvectors = testloader.extractLabelBitVectors(confidencethreshold)

#Create a new csp for each example and assign unary potentials according to the classifier
#Solve the csp using backtracking search
#Compare the resulting assignment to the goldlabel vectors to get accuracy

predictedvectors = {'sentiment':[], 'event':[], 'time':[]}
numtrainingexamples = testX.shape[0]
for exampleindex in range(numtrainingexamples):
	#print 'index: ', exampleindex
	examplecsp = CSP()
	#Load in variables and unary potentials for each labeltype
	for labeltype in ['sentiment', 'event', 'time']:
		numlabels = len(loader.labelnames[labeltype]) 
		# get the unary probabilies for a given example index- sorted arithmetically so should be in order of increasing label index
		labelprobabilities= classifiers[labeltype].predict_proba(testX[exampleindex])# testx needs to be in the correct format
		for labelindex in range(numlabels):
			# get the name of the variable in the csp
			varname = loader.labelnames[labeltype][labelindex]
			#only add variable to csp if it's probability of occuring is nontrivial to save memory
			if labelprobabilities[0][labelindex] > probabilitythreshold:
				# add a variable with domain 1 or 0 representing whether we want to classify this z
				examplecsp.add_variable(varname, [0,1])
				score = labelprobabilities[0][labelindex]
				examplecsp.add_unary_potential(varname, lambda x: math.pow(score, x) * math.pow(1-score, 1-x))
	#add_binary_constraints(binaryConstraints, examplecsp)
	#solve the current example csp 
	backsearch.solve(examplecsp, True, True, True)#gives memory error for full set of labels, need to switch to actual library
	optimalAssignment = Counter(backsearch.optimalAssignment)	

	#Get the predicted vectors from the backtrackingsearch output for each labeltype
	for labeltype in [ 'sentiment', 'event', 'time']:
		labelvector = []
		for labelname in loader.labelnames[labeltype]:
			labelvector.append(optimalAssignment[labelname])
		predictedvectors[labeltype].append(labelvector)

#Perform evaluation on resulting predicted label vector and gold label vector
print 'evaluating results'
evaluator = evaluation.Evaluator()
for labeltype in ['sentiment', 'event', 'time']:
	print 'average rms error for', labeltype, ' : ' , evaluator.rmse(predictedvectors[labeltype], testgoldvectors[labeltype])
	print 'average error for', labeltype, ' : ' ,  evaluator.error_rate(predictedvectors[labeltype], testgoldvectors[labeltype])


