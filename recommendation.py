import os
import sys

if len(sys.argv) < 1:
    print "Usage: recommendation.py tweetsToBeTagged_filename"
    print
    print "Finds similarly classified weather tweets"
    sys.exit()

from sklearn.naive_bayes import MultinomialNB
#import evaluation
from feature_extraction import dataloader
# import ast
# from sklearn.cross_validation import KFold
import time
import vectorToLabel


labeltypes = ['sentiment', 'event', 'time']

#load the raw train csv file into the loader
#TODO: load a separate train and held-out test file into separate loaders
loader = dataloader.DataLoader('data/train.csv')
# get the count matrix representing the wordcount of each word for each example
trained_vectorizer = loader.extractNBCountMatrixFittedVectorizer()
trainX = trained_vectorizer.fit_transform(loader.corpus)

# the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
confidencethreshold = .5
# get the label vectors in the form of bit strings ( one bit string for each label type for each examples)
trainlabelbitstrings = loader.extractLabelBitStrings(confidencethreshold)


classifiers = {}
# train a multinomial naive bayes classifier for each label class 
for labeltype in labeltypes:
	nbclassifier = MultinomialNB()
	#the training label bit string for the given label class
	y = trainlabelbitstrings[labeltype]
	#training the classifier on examples and labels
	nbclassifier.fit(trainX, y)
	#putting the classifier in the dictionary
	classifiers[labeltype] = nbclassifier




print '\n************* Finding similarly labeled tweets for input tweet %s *************\n', 
def load_tweets():
	tweetsToBeTagged_filename = sys.argv[1]
	tweetsToBeTagged = []
	f = open(tweetsToBeTagged_filename)
	tweetsToBeTagged = f.readlines()
	f.close()
	return tweetsToBeTagged

def predictTweets(tweetsToBeTagged):
	print 'Making predictions and converting prediction bitstrings to bitvectors...'
	start_time = time.time()
	testX = trained_vectorizer.transform(tweetsToBeTagged)
	# print 'len(testX) = ', len(testX)
	predictions_list = []
	for testX_matrixcounts in testX:
		predicted_fullbitvector = []
		for labeltype in ['sentiment', 'event', 'time']:
			predicted_label = classifiers[labeltype].predict(testX_matrixcounts)[0]
			predicted_labeltype_bitvector = loader.bitstringToIntList(predicted_label)
			predicted_fullbitvector += predicted_labeltype_bitvector
		predictions_list.append(predicted_fullbitvector)
	elapsed_time = time.time() - start_time
	print 'Completed making predictions and converting prediction bitstrings to bitvectors... took ', elapsed_time
	return predictions_list


def getSimilarTweets(fulllabel_vector):
	pass

tweetsToBeTagged = load_tweets()
predictions_list = predictTweets(tweetsToBeTagged)
converter = vectorToLabel.Converter()

for i in range(len(tweetsToBeTagged)):
	print 'For tweet: %s' %(tweetsToBeTagged[i])
	print '\tPredicted: %s' %(predictions_list[i])
	for labeltype in labeltypes:
		labels = converter.convertToLabels(predictions_list[i])
		print '\tPredicted %s labels: %s' %(labeltype, labels[labeltype])

