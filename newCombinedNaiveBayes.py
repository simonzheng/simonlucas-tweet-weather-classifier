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
			for train_indices, test_indices in kf:
				# get training tweets and gold
				self.training_tweets = [loader.corpus[train_idx] for train_idx in train_indices]
				self.training_gold = [self.gold_bitvectors[train_idx] for train_idx in train_indices]
				
				# fit vectorizer
				self.vectorizer = loader.extractNBCountMatrixFittedVectorizer()
				self.featurized_training_tweets = self.vectorizer.fit_transform(self.training_tweets)
				# print self.vectorizer.get_feature_names()

				# get test tweets and gold
				self.test_tweets = [loader.corpus[test_idx] for test_idx in test_indices]
				self.test_gold = [self.gold_bitvectors[test_idx] for test_idx in test_indices]

				# Get classifiers with the featurized training tweets from vectorizer
				self.classifiers = self.getClassifiers()
				self.evaluateonTest(self.test_tweets, self.test_gold)


	def evaluateonTest(self, test_tweets, test_gold):
		evaluator = evaluation.Evaluator()

		predicted_bitvectors = []
		for tweet in test_tweets:
			# print tweet
			predicted_bitvector = self.combined_classify(tweet)
			# converter.printLabels(predicted_bitvector)
			predicted_bitvectors.append(predicted_bitvector)
		# print test_set
		gold_bitvectors = test_gold

		# # All the incorrect/correct guesses statements
		# evaluator.show_errors(test_tweets, predicted_bitvectors, gold_bitvectors)
		# evaluator.show_correct(test_tweets, predicted_bitvectors, gold_bitvectors)

		rmse = evaluator.rmse(predicted_bitvectors, gold_bitvectors)
		print 'rmse:', rmse

		absolute_accuracy = evaluator.absolute_accuracy(predicted_bitvectors, gold_bitvectors)
		print 'absolute_accuracy', absolute_accuracy


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


	def tweet_features(self, tweet):
		featurized_tweet = self.vectorizer.transform([tweet])
		# print featurized_tweet
		return featurized_tweet

	def combined_classify(self, tweet):
		featurized_tweet = self.tweet_features(tweet)
		return [self.classifiers[i].predict(featurized_tweet) for i in range(self.totalNumLabels)]


converter = vectorToLabel.Converter()
nbc = combinedNBClassifier(data_filename='data/train.csv', numFolds=5)

# def predictUsingNBC(nbc, test_filename):
# 	loader = dataloader.DataLoader(test_filename)
# 	print '******** Predicting now! ********'
# 	test_tweets = loader.corpus
# 	test_gold = loader.extractFullLabelBitVectors(event_label_threshold)
# 	nbc.evaluateonTest(test_tweets, test_gold)
# predictUsingNBC(nbc, 'data/test_5.csv')


