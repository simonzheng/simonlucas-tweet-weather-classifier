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

class combinedNBClassifier:
	def __init__(self, data_filename, numFolds=0):
		# Loading the data
		print 'data comes from file: %s and numFolds is %i' %(data_filename, numFolds)
		loader = dataloader.DataLoader(data_filename)
		self.sentiment_label_indices = range(0,5)
		self.time_label_indices = range(5,9)
		self.event_label_indices = range(9, 24)
		self.label_types = ['sentiment', 'time', 'event']
		
		self.gold_bitvectors = loader.extractFullLabelBitVectors(event_label_threshold)

		# # Test Code to Print if we're extracting correctly
		# converter = vectorToLabel.Converter()
		# for i in range(loader.numDataPoints):
		# 	print '************************************'
		# 	print 'training tweet = %s' %(loader.corpus[i])
		# 	print '\tbitvector is %s' %(self.gold_bitvectors[i])
		# 	converter.printLabels(self.gold_bitvectors[i])

		# Get Useful Constants
		self.totalNumLabels = loader.totalNumLabels
		self.numDataPoints = loader.numDataPoints

		if numFolds == 0:
			# Convert all training data to features and labels
			self.training_tweets = loader.corpus
			self.training_gold = self.gold_bitvectors

			self.vectorizer = loader.extractNBCountMatrixFittedVectorizer()
			self.featurized_training_tweets = self.vectorizer.fit_transform(self.training_tweets)
			# print self.vectorizer.get_feature_names()
			self.classifiers = self.getClassifiers()
		else:
			kf = KFold(loader.numDataPoints, n_folds=numFolds, indices=True)
			all_rmse, all_rmse_by_class = [], []
			all_abs_acc, all_abs_acc_by_class = [], []
			for train_indices, test_indices in kf:
				start_time = time.time()
				# get training tweets and gold
				self.training_tweets = [loader.corpus[train_idx] for train_idx in train_indices]
				self.training_gold = [self.gold_bitvectors[train_idx] for train_idx in train_indices]
				
				# fit vectorizer
				self.vectorizer = loader.extractNBCountMatrixFittedVectorizer()
				self.featurized_training_tweets = self.vectorizer.fit_transform(self.training_tweets)
				# print 'finished fitting vectorizer and fitting training_tweets after this many seconds', time.time() - start_time
				# print self.vectorizer.get_feature_names()

				# get test tweets and gold
				self.test_tweets = [loader.corpus[test_idx] for test_idx in test_indices]
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
		classifiers = []
		for label_index in range(self.totalNumLabels):
			x = self.featurized_training_tweets
			y = [gold_vec[label_index] for gold_vec in self.training_gold]
			classifier = MultinomialNB().fit(x, y)
			classifiers.append(classifier)
		print 'returning %i classifiers' %(len(classifiers))
		return classifiers

	# def getTrainedClassifiersByLabelType(self):
	# 	classifiers = {'sentiment':None, 'time':None, 'event':None}
	# 	# Get Sentiment classifier

	# 	# sentiment
	# 	start_time = time.time()
	# 	sentimentFeaturesAndLabels = [(self.featurized_training_tweets[i], tuple(self.gold_bitvectors[i][0:5])) 
	# 							for i in range(self.numDataPoints)]
	# 	classifier = nltk.NaiveBayesClassifier.train(sentimentFeaturesAndLabels)
	# 	classifiers['sentiment'] = classifier
	# 	elapsed_time = time.time() - start_time
	# 	print 'finished training classifier for sentiment and it took %f seconds' %(elapsed_time)

	# 	# time 
	# 	start_time = time.time()
	# 	timeFeaturesAndLabels = [(self.featurized_training_tweets[i], tuple(self.gold_bitvectors[i][5:9])) 
	# 							for i in range(self.numDataPoints)]
	# 	classifier = nltk.NaiveBayesClassifier.train(timeFeaturesAndLabels)
	# 	elapsed_time = time.time() - start_time
	# 	classifiers['time'] = classifier
	# 	print 'finished training classifier for time and it took %f seconds' %(elapsed_time)

	# 	#event 
	# 	classifiers['event'] = []
	# 	for label_index in self.event_label_indices:
	# 		start_time = time.time()
	# 		featuresAndLabels = [(self.featurized_training_tweets[i], self.gold_bitvectors[i][label_index]) 
	# 							for i in range(self.numDataPoints)]
	# 		# elapsed_time = time.time() - start_time
	# 		# print 'finished constructing new features and labels for label_index %i and it took %f seconds ' %(label_index, elapsed_time)
	# 		train_set = featuresAndLabels
	# 		classifier = nltk.NaiveBayesClassifier.train(train_set)
	# 		classifiers['event'].append(classifier)
	# 		elapsed_time = time.time() - start_time
	# 		print 'finished training classifier for label_index %i and it took %f seconds' %(label_index, elapsed_time)
	# 	print 'returning %i classifiers' %(len(classifiers))
	# 	# print classifiers
	# 	return classifiers

	def combined_classify_tweets(self, tweets):
		numTweets = len(tweets)
		featurized_tweets = self.vectorizer.transform(tweets)
		predicted_labels_by_label = [self.classifiers[i].predict(featurized_tweets) for i in range(self.totalNumLabels)]

		predicted_labels_by_tweet = []
		for tweet_idx in range(numTweets):
			predicted_label = [predicted_labels_by_label[label_idx][tweet_idx] \
							 for label_idx in range(self.totalNumLabels)]
			predicted_labels_by_tweet.append(predicted_label)
		return predicted_labels_by_tweet

converter = vectorToLabel.Converter()
nbc = combinedNBClassifier(data_filename='data/train.csv', numFolds=5)

# def predictUsingNBC(nbc, test_filename):
# 	loader = dataloader.DataLoader(test_filename)
# 	print '******** Predicting now! ********'
# 	test_tweets = loader.corpus
# 	test_gold = loader.extractFullLabelBitVectors(event_label_threshold)
# 	nbc.evaluateOnTest(test_tweets, test_gold)
# predictUsingNBC(nbc, 'data/test_100.csv')


