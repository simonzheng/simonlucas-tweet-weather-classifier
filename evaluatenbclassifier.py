import numpy as np
from sklearn.naive_bayes import MultinomialNB
#import evaluation
from feature_extraction import dataloader
import ast

#load the raw train csv file into the loader
#TODO: load a separate train and held-out test file into separate loaders
loader = dataloader.DataLoader('data/train.csv')
# get the count matrix representing the wordcount of each word for each example
trainX = loader.extractNBCountMatrix() 
# the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
# TODO: look at how varying the confidence threshold affects the accuracy
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
	print "Training Accuracy for " + labeltype + " labels: "
	print accuracy

#TODO: evaluate the test accuracy using a held out test set in a similar way as above



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
