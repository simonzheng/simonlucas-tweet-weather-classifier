import nltk
from feature_extraction import dataloader
import sys
import vectorToLabel

class combinedNBClassifier:
	def __init__(self):
		# Loading the data
		loader = dataloader.DataLoader('data/train.csv')
		event_label_threshold = 1.0/3
		self.gold_bitvectors = loader.extractFullLabelBitVectors(event_label_threshold)

		# # Test Code to Print if we're extracting correctly
		# converter = vectorToLabel.Converter()
		# for i in range(loader.numDataPoints):
		# 	print '************************************'
		# 	print 'training tweet = %s' %(loader.corpus[i])
		# 	print '\tbitvector is %s' %(self.gold_bitvectors[i])
		# 	converter.printLabels(self.gold_bitvectors[i])

		# Create unigram word features (gets 2000 most common tweet words - lowercased)
		all_unigram_words = nltk.FreqDist(loader.getAllWords())
		self.word_features = all_unigram_words.keys()[:2000]

		# Create bigram word features (gets 2000 most common bigram words - lowercased)
		all_bigram_words = nltk.FreqDist(loader.getAllBigramWords())
		self.bigram_features = all_bigram_words.keys()[:2000]

		# Get Useful Constants
		self.totalNumLabels = loader.totalNumLabels
		self.numDataPoints = loader.numDataPoints

		# Convert all training data to features and labels
		self.featurized_tweets = [self.tweet_features(tweet) for tweet in loader.corpus]
		self.classifiers = self.getTrainedClassifiers()

	def getTrainedClassifiers(self):
		classifiers = []
		for label_index in range(self.totalNumLabels):
			featuresAndLabels = [(self.featurized_tweets[i], self.gold_bitvectors[i][label_index]) 
								for i in range(self.numDataPoints)]
			train_set = featuresAndLabels
			classifier = nltk.NaiveBayesClassifier.train(train_set)
			classifiers.append(classifier)
			print 'finished training classifier for label_index %i' %(label_index)
		print 'returning %i classifiers' %(len(classifiers))
		return classifiers

	def tweet_features(self, tweet):
	    tweet_words = set()
	    tweet_bigrams = set()
	    prevWord = '<START>'
	    for word in tweet.split():
	    	word = word.lower()
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

	# def getTrainedClassifiers(self):
	# 	classifiers = {'sentiment' : [], 'time' : [], 'event' : []}
	# 	for labeltype in loader.labeltypes:
	# 		for labelname in loader.labelnames[labeltype]:
	# 			label_index = loader.ordered_keys.index(labelname)

	# 			# Featurize all data
	# 			featuresAndLabels = [(self.featurized_tweets[i], gold_bitvectors[i][label_index]) 
	# 								for i in range(len(loader.numDataPoints))]
	# 			train_set = featuresAndLabels
	# 			classifier = nltk.NaiveBayesClassifier.train(train_set)
	# 	return classifiers

	def combined_classify(self, featurized_tweet):
		return [self.classifiers[i].classify(featurized_tweet) for i in range(self.totalNumLabels)]

	def combined_accuracy(self, test_set):
		test_set_each_label = []
		for i in range(self.totalNumLabels):
			label_test_set = [(test_example[0], test_example[1][i]) for test_example in test_set]
			test_set_each_label.append(label_test_set)
		return [nltk.classify.accuracy(self.classifiers[i], test_set_each_label[i]) for i in range(self.totalNumLabels)]

	def combined_show_most_informative_features(num_features):
		return [self.classifiers[i].show_most_informative_features(num_features) for i in range(self.totalNumLabels)]
		



converter = vectorToLabel.Converter()
nbc = combinedNBClassifier()
loader = dataloader.DataLoader('data/test_100.csv')

print '******** Predicting now! ********'
for tweet in loader.corpus:
	print tweet
	predicted_bitvector = nbc.combined_classify(nbc.tweet_features(tweet))
	converter.printLabels(predicted_bitvector)
gold_bitvectors = loader.extractFullLabelBitVectors(1.0/3)
test_set = [(nbc.tweet_features(loader.corpus[i]), gold_bitvectors[i]) for i in range(loader.numDataPoints)]
# print test_set
print nbc.combined_accuracy(test_set)






