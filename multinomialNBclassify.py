import os
import sys
from feature_extraction import dataloader

#load the raw train csv file into the loader
loader = dataloader.DataLoader('data/train.csv')
import evaluation

def load_raw_tweets_from_file(filename):
	tweetsToBeTagged = []
	f = open(filename)
	tweetsToBeTagged = f.readlines()
	f.close()
	return tweetsToBeTagged

# Checking inputs to load which tweet to check
if len(sys.argv) < 1:
    print "Usage: multinomialNBclassify.py [-f tweetsToBeTagged_filename] <<OR>> [-i start_index end_index exclusive]"
    sys.exit()

start_index, end_index = None, None
if sys.argv[1] == '-f':
	file_name = sys.argv[2]
	tweetsToBeTagged = load_raw_tweets_from_file(file_name)
elif sys.argv[1] == '-i':
	start_index = int(sys.argv[2])
	end_index = int(sys.argv[3])
	tweetsToBeTagged = loader.corpus[start_index : end_index]
else:
	print "Usage: compareClassifiers.py [-f tweetsToBeTagged_filename] <<OR>> [-i start_index end_index]"
	sys.exit()


from sklearn.naive_bayes import MultinomialNB
#import evaluation
# import ast
# from sklearn.cross_validation import KFold
import time
import vectorToLabel


labeltypes = ['sentiment', 'event', 'time']

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


print '\n******************************************'
print 'Get gold set if we use -i option for indices [%i:%i]' %(start_index, end_index)
gold_list = None
if (start_index != None and end_index != None):
	test_indices = range(start_index, end_index)	
	labelDict = loader.extractLabelBitVectors(confidencethreshold, indices=test_indices)	
	gold_list = loader.combineLabelBitVectors(labelDict)
print '******************************************'
predictions_list = predictTweets(tweetsToBeTagged)
converter = vectorToLabel.Converter()

for i in range(len(tweetsToBeTagged)):
	print '******************************************'
	print 'Tweet: %s' %(tweetsToBeTagged[i])
	if gold_list != None:
		print '******************************************'
		print '\tActual bitvector: '
		converter.printLabels(gold_list[i])
	print '******************************************'
	print '\tPredicted bitvector: '
	converter.printLabels(predictions_list[i])

print '******************************************'
evaluator = evaluation.Evaluator()
rmse = evaluator.rmse(predictions_list, gold_list)
print 'rmse is %f' %rmse
print '******************************************'

