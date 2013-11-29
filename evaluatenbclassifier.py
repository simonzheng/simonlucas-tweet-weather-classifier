import numpy as np
from sklearn.naive_bayes import MultinomialNB
#import evaluation
from feature_extraction import dataloader
import ast
from sklearn.cross_validation import KFold

print '\n************* Evaluating our Naive Bayes Classifier on Training Data *************\n'

labeltypes = ['sentiment', 'event', 'time']

#load the raw train csv file into the loader
#TODO: load a separate train and held-out test file into separate loaders
loader = dataloader.DataLoader('data/train.csv')
# get the count matrix representing the wordcount of each word for each example
trainX = loader.extractNBCountMatrix()
# the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
confidencethreshold = .5
# get the label vectors in the form of bit strings ( one bit string for each label type for each examples)
trainlabelbitstrings = loader.extractLabelBitStrings(confidencethreshold) 

classifiers = {}
# train a multinomial naive bayes classifier for each label class 
for labeltype in ['sentiment', 'event', 'time']:
	nbclassifier = MultinomialNB()
	#the training label bit string for the given label class
	y = trainlabelbitstrings[labeltype]
	#training the classifier on examples and labels
	nbclassifier.fit(trainX, y)
	#putting the classifier in the dictionary
	classifiers[labeltype] = nbclassifier

#evaluate the training accuracy of the classifier for each label class using the internal MultinomialNB score method 
for labeltype in ['sentiment', 'event', 'time']:
	accuracy = classifiers[labeltype].score(trainX, trainlabelbitstrings[labeltype])
	print "Training Accuracy for " + labeltype + " labels: ", accuracy

######## K Means Cross-Validation with k=5 held-out test sets ###########
print '\n************* Evaluating our Naive Bayes Classifier on Test Data *************\n'
loader = dataloader.DataLoader('data/train.csv')
numFolds = 5
kf = KFold(loader.numDataPoints, n_folds=numFolds, indices=True)

fold_count = 1
all_accuracies = {}
for labeltype in labeltypes:
	all_accuracies[labeltype] = []
for train_indices, test_indices in kf:
	print 'Currently performing k-fold validation on fold iteration #%d' %(fold_count)
	print('Train: %s; Test: %s' %(train_indices, test_indices))
	# get the count matrix representing the wordcount of each word for each example
	trainX, testX = loader.extractTrainingAndTestCountMatrices(training_indices=train_indices)
	# the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
	confidencethreshold = .5
	# get the label vectors in the form of bit strings ( one bit string for each label type for each examples)
	trainlabelbitstrings = loader.extractLabelBitStrings(confidencethreshold, indices=train_indices)
	testlabelbitstrings = loader.extractLabelBitStrings(confidencethreshold, indices=test_indices) 

	classifiers = {}
	# train a multinomial naive bayes classifier for each label class 
	for labeltype in ['sentiment', 'event', 'time']:
		nbclassifier = MultinomialNB()
		#the training label bit string for the given label class
		y = trainlabelbitstrings[labeltype]
		#training the classifier on examples and labels
		nbclassifier.fit(trainX, y)
		#putting the classifier in the dictionary
		classifiers[labeltype] = nbclassifier

	for labeltype in ['sentiment', 'event', 'time']:
		accuracy = classifiers[labeltype].score(testX, testlabelbitstrings[labeltype])
		all_accuracies[labeltype].append(accuracy)
		print "Test Accuracy for " + labeltype + " labels: ", accuracy

	fold_count += 1 # Update the fold count to next iteration

for labeltype in labeltypes:
	print "Overall Test Accuracy for " + labeltype + " labels: ", np.mean(all_accuracies[labeltype])

#POSSIBLE TODO: perform rms error analysis using your module. 
# the output of classifiers[labeltype].predict(testX) is a bit string (required a hashable type)
# so if you can figure out how to convert a bit string to a tuple you can do rms 
# This might give us something interesting to write about in terms of error analysis

#Junk Code that might be resused 
	#predictedvectors = []
	#for example in testX:
	#	predicted = classifiers[labeltype].predict(testX)[0]
	#	predictedvectors.append(predicted)
	#expectedvectors = testY[labeltype][0:100]
	#error_rate = evaluator.error_rate(predictedvectors, expectedvectors)
