import os
import sys
from feature_extraction import dataloader
import random
import nltk

class activityRecommender:
	def __init__(self, data_filename):
		# Loading the data
		print 'data for activityRecommender comes from file: %s' %(data_filename)
		self.loader = dataloader.DataLoader(data_filename)
		# Get Useful Constants
		self.totalNumLabels = self.loader.totalNumLabels
		self.numDataPoints = self.loader.numDataPoints
		self.sentiment_label_indices = range(0,5)
		self.time_label_indices = range(5,9)
		self.event_label_indices = range(9, 24)
		self.label_types = ['sentiment', 'time', 'event']
		event_label_threshold = 1.0/3
		self.gold_bitvectors = self.loader.extractFullLabelBitVectors(event_label_threshold)
		# tagged_corpus_path = os.path.realpath('data/train_tagged.txt')
		# tagged_corp_reader = nltk.corpus.reader.TaggedCorpusReader(os.path.dirname(tagged_corpus_path), os.path.basename(tagged_corpus_path), sep='_')
		# self.tagged_tweets = tagged_corp_reader.tagged_sents()

		candidate_corpus_path = os.path.realpath('data/train_tagged.csv')

		# gs = nltk.corpus.reader.TaggedCorpusReader(os.path.dirname(gs_corpus_path), os.path.basename(gs_corpus_path), sep='_')
		candidate = nltk.corpus.reader.TaggedCorpusReader(os.path.dirname(candidate_corpus_path), os.path.basename(candidate_corpus_path), sep='_')

		self.tagged_tweets = candidate.tagged_sents()

		# # Test Code to Print if we're extracting correctly
		# converter = vectorToLabel.Converter()
		# for i in range(self.loader.numDataPoints):
		# 	print '************************************'
		# 	print 'training tweet = %s' %(self.loader.corpus[i])
		# 	print '\tbitvector is %s' %(self.gold_bitvectors[i])
		# 	converter.printLabels(self.gold_bitvectors[i])

	def getSimilarTweets(self, vectorToMatch):
		similarTweets = []
		numSimilarTweetsToFind = 3

		shuffledCorpus = random.shuffle(self.loader.corpus)

		for tweet_idx in range(len(self.loader.corpus)):
			tweet = self.loader.corpus[tweet_idx]
			corpusTweetLabels = self.gold_bitvectors[tweet_idx]
			if self.checkCriteria(vectorToMatch, corpusTweetLabels):
				verbs = self.getVerbs(tweet_idx)
				similarTweets.append({'tweet': tweet,
										'labels': corpusTweetLabels,
										'verbs': verbs})
		return similarTweets

	def checkCriteria(self, vectorToMatch, corpusTweetLabelVector):
		# check positive
		indexOfPositiveLabel = 3
		if corpusTweetLabelVector[indexOfPositiveLabel] != 1:
			return False

		# check if weather event conditions are similar
		for event_label_idx in self.event_label_indices:
			if vectorToMatch[event_label_idx] != corpusTweetLabelVector[event_label_idx]: 
				return False
		return True

	def getVerbs(self, tweet_idx):
		tagged_tweet = self.tagged_tweets[tweet_idx]
		# print '----------------------------------------------------------------------'
		# print '--> The tagged tweet is: ', tagged_tweet
		verbFound = False
		verbs = []
		for (word, tag) in tagged_tweet:
			if tag != None and tag.startswith('V'):
				# print '\t\tVerb: ', word
				verbs.append(word)
				verbFound = True
		if not verbFound:
			# print '\t\t *** No Verbs Found ***'
			verbs = None
		return verbs


if len(sys.argv) <= 1:
    print "Usage: recommendation.py tweetsToBeTagged_filename"
    sys.exit()

def load_tweets(filename=None):
	tweetsToBeTagged_filename = sys.argv[1]
	if len(tweetsToBeTagged_filename) <= 0:
		print 'tweetsToBeTagged_filename must be at least one char!'
		sys.exit()
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


import vectorToLabel
tweetsToBeTagged = load_tweets()

filename='data/train.csv'

# import combinedNaiveBayes
# cnbc = combinedNaiveBayes.combinedNBClassifier(data_filename='data/train.csv', numFolds=0)
# predictions_list = cnbc.combined_classify_tweets(tweetsToBeTagged)
import structuredNaiveBayes
snbc = structuredNaiveBayes.structuredNBClassifier(data_filename=filename, numFolds=0)
print 'loaded classifier. now predicting'
predictions_list = snbc.combined_classify_tweets(tweetsToBeTagged)
print 'predicted!'

# pickle.write(name of data structure, open(filename, 'wb'))
# prediction_list = pickle.load(open())

converter = vectorToLabel.Converter()
recommender = activityRecommender(filename)
print 'loaded recommender'

for i in range(len(tweetsToBeTagged)):
	prediction_vec = predictions_list[i]
	print '\n*************************************'
	print 'For tweet: %s' %(tweetsToBeTagged[i])
	print '\tPredicted: %s' %(prediction_vec)
	labels = converter.convertToLabels(prediction_vec)
	for labeltype in converter.labeltypes:
		print '\tPredicted %s labels: %s' %(labeltype, labels[labeltype])
	similarTweets = recommender.getSimilarTweets(prediction_vec)

	print '\nLooking at Similar Tweets:'
	for similarTweet in similarTweets:
		# print '\tSimilar Tweet:', similarTweet['tweet']
		# print '\tVerbs:', similarTweet['verbs']
		print '\tPerhaps you should try any of these verbs %s from such as in this similar positive tweets: %s' %(str(similarTweet['verbs']), similarTweet['tweet'])

