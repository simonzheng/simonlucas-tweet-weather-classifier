import nltk
from feature_extraction import dataloader
import sys
import vectorToLabel
import time
import string
from sklearn.cross_validation import KFold
import evaluation

class combinedNBClassifier:
	def __init__(self, training_filename = 'data/test_5.csv', eachlabel=True):
		self.eachlabel = eachlabel
		# Loading the data
		start_time = time.time()

		print 'training with file: %s' %(training_filename)
		
		loader = dataloader.DataLoader(training_filename)
		event_label_threshold = 1.0/3
		self.event_label_indices = range(9, 24)
		self.sentiment_label_indices = range(0,5)
		self.time_label_indices = range(5,9)

		self.gold_bitvectors = loader.extractFullLabelBitVectors(event_label_threshold)

		# # Test Code to Print if we're extracting correctly
		# converter = vectorToLabel.Converter()
		# for i in range(loader.numDataPoints):
		# 	print '************************************'
		# 	print 'training tweet = %s' %(loader.corpus[i])
		# 	print '\tbitvector is %s' %(self.gold_bitvectors[i])
		# 	converter.printLabels(self.gold_bitvectors[i])

		# Create unigram word features (gets x most common tweet words - lowercased)
		all_unigram_words = nltk.FreqDist(loader.getAllWords())
		self.word_features = all_unigram_words.keys()[:2000]
		print all_unigram_words

		# Create bigram word features (gets x most common bigram words - lowercased)
		all_bigram_words = nltk.FreqDist(loader.getAllBigramWords())
		self.bigram_features = all_bigram_words.keys()[:1000]
		print all_bigram_words

		# Get Useful Constants
		self.totalNumLabels = loader.totalNumLabels
		self.numDataPoints = loader.numDataPoints

		# Convert all training data to features and labels
		self.featurized_tweets = [self.tweet_features(tweet) for tweet in loader.corpus]

		elapsed_time = time.time() - start_time
		print 'finished init before classifiers and it took %f seconds ' %(elapsed_time)

		self.classifiers = None
		if eachlabel == False:
			self.classifiers = self.getTrained24Classifiers()
		else:
			self.classifiers = self.getTrainedClassifiersByLabelType()

	def getTrained24Classifiers(self):
		classifiers = []
		for label_index in range(self.totalNumLabels):
			start_time = time.time()
			featuresAndLabels = [(self.featurized_tweets[i], self.gold_bitvectors[i][label_index]) 
								for i in range(self.numDataPoints)]
			# elapsed_time = time.time() - start_time
			# print 'finished constructing new features and labels for label_index %i and it took %f seconds ' %(label_index, elapsed_time)
			train_set = featuresAndLabels
			classifier = nltk.NaiveBayesClassifier.train(train_set)
			classifiers.append(classifier)
			elapsed_time = time.time() - start_time
			print 'finished training classifier for label_index %i and it took %f seconds' %(label_index, elapsed_time)
		print 'returning %i classifiers' %(len(classifiers))
		return classifiers

	def getTrainedClassifiersByLabelType(self):
		classifiers = {'sentiment':None, 'time':None, 'event':None}
		# Get Sentiment classifier

		# sentiment
		start_time = time.time()
		sentimentFeaturesAndLabels = [(self.featurized_tweets[i], tuple(self.gold_bitvectors[i][0:5])) 
								for i in range(self.numDataPoints)]
		classifier = nltk.NaiveBayesClassifier.train(sentimentFeaturesAndLabels)
		classifiers['sentiment'] = classifier
		elapsed_time = time.time() - start_time
		print 'finished training classifier for sentiment and it took %f seconds' %(elapsed_time)

		# time 
		start_time = time.time()
		timeFeaturesAndLabels = [(self.featurized_tweets[i], tuple(self.gold_bitvectors[i][5:9])) 
								for i in range(self.numDataPoints)]
		classifier = nltk.NaiveBayesClassifier.train(timeFeaturesAndLabels)
		elapsed_time = time.time() - start_time
		classifiers['time'] = classifier
		print 'finished training classifier for time and it took %f seconds' %(elapsed_time)

		#event 
		classifiers['event'] = []
		for label_index in self.event_label_indices:
			start_time = time.time()
			featuresAndLabels = [(self.featurized_tweets[i], self.gold_bitvectors[i][label_index]) 
								for i in range(self.numDataPoints)]
			# elapsed_time = time.time() - start_time
			# print 'finished constructing new features and labels for label_index %i and it took %f seconds ' %(label_index, elapsed_time)
			train_set = featuresAndLabels
			classifier = nltk.NaiveBayesClassifier.train(train_set)
			classifiers['event'].append(classifier)
			elapsed_time = time.time() - start_time
			print 'finished training classifier for label_index %i and it took %f seconds' %(label_index, elapsed_time)
		print 'returning %i classifiers' %(len(classifiers))
		# print classifiers
		return classifiers


	def tweet_features(self, tweet):
	    tweet_words = set()
	    tweet_bigrams = set()
	    prevWord = '<START>'
	    for word in tweet.split():
	    	word = word.lower().translate(string.maketrans("",""), string.punctuation)
	    	tweet_words.add(word)
	    	bigram = (prevWord, word)
	    	tweet_bigrams.add(bigram)
	    	prevWord = word

	    features = {}
	    for word in self.word_features:
	        features['unigram(%s)' % word] = (word in tweet_words)
	    for bigram in self.bigram_features:
	        features['bigram(%s)' % str(bigram)] = (bigram in tweet_bigrams)
	    return features

	def combined_classify(self, tweet):
		featurized_tweet = self.tweet_features(tweet)
		if self.eachlabel == True:
			return self.combined_classify_each_labeltype(featurized_tweet)
		else:
			return self.combined_classify_with24(featurized_tweet)

	def combined_classify_with24(self, featurized_tweet):
		return [self.classifiers[i].classify(featurized_tweet) for i in range(self.totalNumLabels)]

	def combined_classify_each_labeltype(self, featurized_tweet):
		prediction = list(self.classifiers['sentiment'].classify(featurized_tweet)) + \
				list(self.classifiers['time'].classify(featurized_tweet)) + \
				[classifier.classify(featurized_tweet) for classifier in self.classifiers['event']]
		return prediction

	# # doesn't work for new classification
	# def combined_accuracy(self, test_set):
	# 	test_set_each_label = []
	# 	for i in range(self.totalNumLabels):
	# 		label_test_set = [(test_example[0], test_example[1][i]) for test_example in test_set]
	# 		test_set_each_label.append(label_test_set)
	# 	return [nltk.classify.accuracy(self.classifiers[i], test_set_each_label[i]) for i in range(self.totalNumLabels)]

	def combined_show_most_informative_features(self, num_features):
		if self.eachlabel == True:
			self.combined_show_most_informative_features_each_labeltype(num_features)
		else:
			self.combined_show_most_informative_features_with24(num_features)

	def combined_show_most_informative_features_with24(self, num_features):
		for i in range(self.totalNumLabels):
			print 'for index ', i
			self.classifiers[i].show_most_informative_features(num_features)

	def combined_show_most_informative_features_each_labeltype(self, num_features):
		print 'for sentiment:'
		self.classifiers['sentiment'].show_most_informative_features(num_features)

		print 'for time'
		self.classifiers['time'].show_most_informative_features(num_features)

		for event_classifier_index in range(len(self.classifiers['event'])):
			print 'for event k', event_classifier_index+1
			classifier = self.classifiers['event'][event_classifier_index]
			classifier.show_most_informative_features(num_features)






converter = vectorToLabel.Converter()
nbc = combinedNBClassifier(training_filename='data/train.csv', eachlabel=True)
evaluator = evaluation.Evaluator()

def predictUsingNBC(nbc, test_filename):
	loader = dataloader.DataLoader(test_filename)

	print '******** Predicting now! ********'
	predicted_bitvectors = []
	test_tweets = loader.corpus
	for tweet in test_tweets:
		print tweet
		predicted_bitvector = nbc.combined_classify(tweet)
		converter.printLabels(predicted_bitvector)
		predicted_bitvectors.append(predicted_bitvector)
	gold_bitvectors = loader.extractFullLabelBitVectors(1.0/3)
	test_set = [(nbc.tweet_features(loader.corpus[i]), gold_bitvectors[i]) for i in range(loader.numDataPoints)]
	# print test_set
	
	nbc.combined_show_most_informative_features(5)

	evaluator.show_errors(test_tweets, predicted_bitvectors, gold_bitvectors)
	# evaluator.show_correct(test_tweets, predicted_bitvectors, gold_bitvectors)

	rmse = evaluator.rmse(predicted_bitvectors, gold_bitvectors)
	print 'rmse:', rmse

	absolute_accuracy = evaluator.absolute_accuracy(predicted_bitvectors, gold_bitvectors)
	print 'absolute_accuracy', absolute_accuracy



predictUsingNBC(nbc, 'data/test_100.csv')


