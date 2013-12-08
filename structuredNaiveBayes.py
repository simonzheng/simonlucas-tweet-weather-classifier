import nltk
from feature_extraction import dataloader
import sys
import vectorToLabel
import time
import string
from sklearn.cross_validation import KFold
import evaluation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

event_label_threshold = 1.0/3

class structuredNBClassifier:
	def __init__(self, data_filename, numFolds=0):
		# Loading the data
		print 'data comes from file: %s and numFolds is %i' %(data_filename, numFolds)
		self.loader = dataloader.DataLoader(data_filename)
		# Get Useful Constants
		self.totalNumLabels = self.loader.totalNumLabels
		self.numDataPoints = self.loader.numDataPoints
		self.sentiment_label_indices = range(0,5)
		self.time_label_indices = range(5,9)
		self.event_label_indices = range(9, 24)
		self.label_types = ['sentiment', 'time', 'event']
		# get the label vectors in the form of bit strings ( one bit string for each label type for each examples)

		self.gold_bitvectors = self.loader.extractFullLabelBitVectors(event_label_threshold)

		# # Test Code to Print if we're extracting correctly
		# converter = vectorToLabel.Converter()
		# for i in range(self.loader.numDataPoints):
		# 	print '************************************'
		# 	print 'training tweet = %s' %(self.loader.corpus[i])
		# 	print '\tbitvector is %s' %(self.gold_bitvectors[i])
		# 	converter.printLabels(self.gold_bitvectors[i])

		if numFolds == 0:
			# Convert all training data to features and labels
			self.train_indices = None
			self.training_tweets = self.loader.corpus
			self.training_gold = self.gold_bitvectors

			self.vectorizer = self.loader.extractNBCountMatrixFittedVectorizer()
			self.featurized_training_tweets = self.vectorizer.fit_transform(self.training_tweets)
			# print self.vectorizer.get_feature_names()
			self.classifiers = self.getClassifiers()
		else:
			kf = KFold(self.loader.numDataPoints, n_folds=numFolds, indices=True)
			all_rmse, all_rmse_by_class = [], []
			all_abs_acc, all_abs_acc_by_class = [], []
			for train_indices, test_indices in kf:
				print 'performing kf tesing on ', test_indices
				self.train_indices = train_indices
				start_time = time.time()
				# get training tweets and gold
				self.training_tweets = [self.loader.corpus[train_idx] for train_idx in train_indices]
				self.training_gold = [self.gold_bitvectors[train_idx] for train_idx in train_indices]
				
				# fit vectorizer
				self.vectorizer = self.loader.extractNBCountMatrixFittedVectorizer()
				self.featurized_training_tweets = self.vectorizer.fit_transform(self.training_tweets)
				# print 'finished fitting vectorizer and fitting training_tweets after this many seconds', time.time() - start_time
				# print self.vectorizer.get_feature_names()

				# get test tweets and gold
				self.test_tweets = [self.loader.corpus[test_idx] for test_idx in test_indices]
				self.test_gold = [self.gold_bitvectors[test_idx] for test_idx in test_indices]

				# Get classifiers with the featurized training tweets from vectorizer
				self.classifiers = self.getClassifiers()
				
				# print 'finished getting classifiers after this many seconds:', time.time() - start_time
				
				rmse, rmse_by_class, absolute_accuracy, absolute_accuracy_by_class = self.evaluateOnTest(self.test_tweets, self.test_gold)
				# print 'finished getting classifiers after this many seconds', time.time() - start_time
				all_rmse.append(rmse)
				all_rmse_by_class.append(rmse_by_class)
				all_abs_acc.append(absolute_accuracy)
				all_abs_acc_by_class.append(absolute_accuracy_by_class)

			print 'overall rmse = ', np.mean(all_rmse)
			print 'overall rmse by class = '
			for label_type in self.label_types:
				print '\t', label_type, np.mean([fold_rmse[label_type] for fold_rmse in all_rmse_by_class])
			print 'overall absolute_accuracy = ', np.mean(all_abs_acc)
			print 'overall absolute_accuracy_by_class = ' 
			for label_type in self.label_types:
				print '\t', label_type, np.mean([fold_rmse[label_type] for fold_rmse in all_abs_acc_by_class])



	def evaluateOnTest(self, test_tweets, test_gold):
		evaluator = evaluation.Evaluator()
		predicted_bitvectors = self.combined_classify_tweets(test_tweets)
		# converter.printLabels(predicted_bitvector)
		# print test_set
		gold_bitvectors = test_gold

		# # All the incorrect/correct guesses statements
		# evaluator.show_errors(test_tweets, predicted_bitvectors, gold_bitvectors)
		# evaluator.show_correct(test_tweets, predicted_bitvectors, gold_bitvectors)

		rmse = evaluator.rmse(predicted_bitvectors, gold_bitvectors)
		print 'rmse:', rmse
		rmse_by_class = evaluator.rmse_by_labelclass(predicted_bitvectors, gold_bitvectors)
		print 'rmse_by_class', rmse_by_class

		absolute_accuracy = evaluator.absolute_accuracy(predicted_bitvectors, gold_bitvectors)
		print 'absolute_accuracy', absolute_accuracy

		absolute_accuracy_by_class = evaluator.absolute_accuracy_by_labelclass(predicted_bitvectors, gold_bitvectors)
		print 'absolute_accuracy_by_class', absolute_accuracy_by_class

		return rmse, rmse_by_class, absolute_accuracy, absolute_accuracy_by_class


	def getClassifiers(self):
		trainlabelbitstrings_byclass = self.loader.extractLabelBitStrings(event_label_threshold, self.train_indices)
		classifiers = {}

		# train a multinomial naive bayes classifier for each label class 
		for labeltype in self.label_types:
			nbclassifier = MultinomialNB()
			#the training label bit string for the given label class
			x = self.featurized_training_tweets
			y = trainlabelbitstrings_byclass[labeltype]
			#training the classifier on examples and labels
			nbclassifier.fit(x, y)
			#putting the classifier in the dictionary
			classifiers[labeltype] = nbclassifier
		print 'returning %i classifiers' %(len(classifiers))
		return classifiers

	def combined_classify_tweets(self, tweets):
		numTweets = len(tweets)
		featurized_tweets = self.vectorizer.transform(tweets)
		predicted_labels_by_label = {}
		for label_type in self.label_types:
			predicted_labels_by_label[label_type] = self.classifiers[label_type].predict(featurized_tweets)

		predicted_labels_by_tweet = []
		for tweet_idx in range(numTweets):
			predicted_label_by_labeltype = [predicted_labels_by_label[label_type][tweet_idx] \
							 for label_type in self.label_types]

			predicted_label = []
			for type_prediction in predicted_label_by_labeltype:
				predicted_label += [int(bit) for bit in type_prediction.split()]
			predicted_labels_by_tweet.append(predicted_label)
		return predicted_labels_by_tweet

# converter = vectorToLabel.Converter()
# nbc = structuredNBClassifier(data_filename='data/train.csv', numFolds=5)

# # Note: use this if already trained but numFolds = 0
# def predictUsingNBC(nbc, test_filename):
# 	loader = dataloader.DataLoader(test_filename)
# 	print '******** Predicting now! ********'
# 	test_tweets = loader.corpus
# 	test_gold = loader.extractFullLabelBitVectors(event_label_threshold)
# 	nbc.evaluateOnTest(test_tweets, test_gold)
# predictUsingNBC(nbc, 'data/test_5.csv')









#########################
##### Old Code Below ####
#########################


# import numpy as np
# from sklearn.naive_bayes import MultinomialNB
# from feature_extraction import dataself.loader
# # import ast
# from sklearn.cross_validation import KFold
# import evaluation
# import time





# labeltypes = ['sentiment', 'event', 'time']
# filename = 'data/test_5.csv'

# #load the raw train csv file into the self.loader
# #TODO: load a separate train and held-out test file into separate self.loaders
# self.loader = dataself.loader.Dataself.Loader(filename)
# # get the count matrix representing the wordcount of each word for each example
# trainX = self.loader.extractNBCountMatrix()
# # the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
# confidencethreshold = .5
# # get the label vectors in the form of bit strings ( one bit string for each label type for each examples)
# trainlabelbitstrings_byclass = self.loader.extractLabelBitStrings(confidencethreshold) 

# classifiers = {}
# # train a multinomial naive bayes classifier for each label class 
# for labeltype in labeltypes:
# 	nbclassifier = MultinomialNB()
# 	#the training label bit string for the given label class
# 	y = trainlabelbitstrings_byclass[labeltype]
# 	#training the classifier on examples and labels
# 	nbclassifier.fit(trainX, y)
# 	#putting the classifier in the dictionary
# 	classifiers[labeltype] = nbclassifier







# print "\n************* Evaluating our Naive Bayes Classifier on Training Data *************\n"
# #evaluate the training accuracy of the classifier for each label class using the internal MultinomialNB score method 
# for labeltype in labeltypes:
# 	accuracy = classifiers[labeltype].score(trainX, trainlabelbitstrings_byclass[labeltype])
# 	print "Training Accuracy for " + labeltype + " labels: ", accuracy

# ######## K Means Cross-Validation with k=5 held-out test sets ###########
# print "\n************* Evaluating our Naive Bayes Classifier on Test Data *************\n"
# self.loader = dataself.loader.Dataself.Loader(filename)
# numFolds = 5
# kf = KFold(self.loader.numDataPoints, n_folds=numFolds, indices=True)


# fold_count = 1
# all_accuracies, all_rmse = {}, {}
# for labeltype in labeltypes:
# 	all_accuracies[labeltype] = []
# 	all_rmse[labeltype] = []
# for train_indices, test_indices in kf:
# 	print 'Currently performing k-fold validation on fold iteration #%d' %(fold_count)
# 	print('Train: %s; Test: %s' %(train_indices, test_indices))
# 	# get the count matrix representing the wordcount of each word for each example
# 	print 'Getting training and test Count Matrices...'
# 	start_time = time.time()
# 	trainX, testX = self.loader.extractTrainingAndTestCountMatrices(training_indices=train_indices)
# 	elapsed_time = time.time() - start_time
# 	print 'Completed Getting training and test Count Matrices... took ', elapsed_time
# 	# the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
# 	confidencethreshold = .5
# 	# get the label vectors in the form of bit strings ( one bit string for each label type for each examples)
# 	print 'Getting trainlabelbitstrings_byclass with the specified indices...'
# 	start_time = time.time()
# 	trainlabelbitstrings_byclass = self.loader.extractLabelBitStrings(confidencethreshold, indices=train_indices)
# 	elapsed_time = time.time() - start_time
# 	print 'Completed getting trainlabelbitstrings_byclass with the specified indices... took ', elapsed_time

# 	print 'Getting testlabelbitvectors with the specified indices...'
# 	start_time = time.time()
# 	testlabelbitvectors = self.loader.extractLabelBitVectors(confidencethreshold, indices=test_indices)
# 	elapsed_time = time.time() - start_time
# 	print 'Completed getting testlabelbitvectors with the specified indices... took ', elapsed_time


# 	print 'Getting testlabelbitstrings with the specified indices...'
# 	start_time = time.time()
# 	testlabelbitstrings = self.loader.extractLabelBitStrings(confidencethreshold, indices=test_indices)
# 	elapsed_time = time.time() - start_time
# 	print 'Completed Getting testlabelbitstrings with the specified indices...', elapsed_time

# 	classifiers = {}
# 	# train a multinomial naive bayes classifier for each label class 
# 	for labeltype in labeltypes:
# 		nbclassifier = MultinomialNB()
# 		#the training label bit string for the given label class
# 		y = trainlabelbitstrings_byclass[labeltype]
# 		#training the classifier on examples and labels
# 		nbclassifier.fit(trainX, y)
# 		#putting the classifier in the dictionary
# 		classifiers[labeltype] = nbclassifier

# 	for labeltype in labeltypes:
# 		# Calculate MultinomialNB accuracy scores
# 		accuracy = classifiers[labeltype].score(testX, testlabelbitstrings[labeltype])
		
# 		# Calculate RMSE scores
# 		print 'Making predictions and converting prediction bitstrings to bitvectors for %s labels...' %(labeltype)
# 		start_time = time.time()
# 		predictions_list = []
# 		for testX_matrixcounts in testX:
# 			predicted_label = classifiers[labeltype].predict(testX_matrixcounts)[0]
# 			predicted_bitvector = self.loader.bitstringToIntList(predicted_label)
# 			predictions_list.append(predicted_bitvector)
		
# 		elapsed_time = time.time() - start_time
# 		print 'Completed converting prediction bitstrings to bitvectors... took ', elapsed_time
# 		gold_list = testlabelbitvectors[labeltype]

# 		# # Sanity check that conversion from bitstring to bitvector was successful
# 		# print 'gold_list[0] = ', gold_list[0]
# 		# print 'first char gold_list[0][0] = ', gold_list[0][0]

# 		# print 'predictions_list[0] = ', predictions_list[0]
# 		# print 'first char predictions_list[0][0] = ', predictions_list[0][0]
		
# 		evaluator = evaluation.Evaluator()
# 		rmse = evaluator.rmse(predictions_list, gold_list)
# 		all_rmse[labeltype].append(rmse)

# 		all_accuracies[labeltype].append(accuracy)
# 		print "Test Accuracy for " + labeltype + " labels: ", accuracy, 'for iteration: ', fold_count
# 		print "Test RMSE for " + labeltype + " labels: ", rmse, 'for iteration: ', fold_count

# 	fold_count += 1 # Update the fold count to next iteration

# for labeltype in labeltypes:
# 	print "Overall Test Accuracy for " + labeltype + " labels: ", np.mean(all_accuracies[labeltype])
# 	print "Overall Test RMSE for " + labeltype + " labels: ", np.mean(all_rmse[labeltype])

# #POSSIBLE TODO: perform rms error analysis using your module. 
# # the output of classifiers[labeltype].predict(testX) is a bit string (required a hashable type)
# # so if you can figure out how to convert a bit string to a tuple you can do rms 
# # This might give us something interesting to write about in terms of error analysis


# #Junk Code that might be resused 
# 	#predictedvectors = []
# 	#for example in testX:
# 	#	predicted = classifiers[labeltype].predict(testX)[0]
# 	#	predictedvectors.append(predicted)
# 	#expectedvectors = testY[labeltype][0:100]
# 	#error_rate = evaluator.error_rate(predictedvectors, expectedvectors)
