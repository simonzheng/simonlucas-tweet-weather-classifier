import csv
import collections
import string
from sklearn.feature_extraction.text import CountVectorizer
# DataLoader loads a dictionary representing the raw csv data 
class DataLoader:
	def __init__(self, testfile):
		self.raw_test_data = self.loadTrainData(testfile)
		self.labelnames = \
			{'sentiment':['s1', 's2', 's3', 's4', 's5'], 
			'time':['w1', 'w2', 'w3', 'w4'],
			'event':['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']
			}
		self.ordered_keys = ['s1', 's2', 's3', 's4', 's5', 
							'w1', 'w2', 'w3', 'w4', 
							'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 
							'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']
		self.labeltypes = ['sentiment', 'time', 'event']
		self.vectorizer = CountVectorizer(ngram_range=(1,2),
			token_pattern=r'\b\w+\b', min_df=1)
		self.corpus = [entry['tweet'] for entry in self.raw_test_data]
		self.numDataPoints = len(self.raw_test_data)
		self.totalNumLabels = len(self.labelnames['sentiment']) + len(self.labelnames['time']) + len(self.labelnames['event'])
		self.event_label_indices = range(10,24)

		self.all_words, self.bigram_words = None, None
 
	# returns the raw data in dictionary form where we can access the tweet with raw_test_data[<entryindex>]['tweet]']
	def loadTrainData(self, filename):
		raw_data = []
		with open(filename, 'rb') as csvfile:
			reader = csv.DictReader(csvfile, delimiter=',')
			for row in reader:
				raw_data.append(row)
		return raw_data

	# returns an array of counters of words and their word counts
	def extractFeatureVectors(self):
		features = []
		for entry in self.raw_test_data:
			text = entry["tweet"]
			features.append(collections.Counter(text.split(' ')))
		return features

	# Old Verson that works with structureprediction implementation
	def extractStructuredNBCountMatrix(self):
		X = self.vectorizer.fit_transform(self.corpus)
		return X

	# returns a matrix representing word count that can be used to train a scikit classifier
	def extractNBCountMatrix(self, indices=None):
		vectorizer = CountVectorizer(ngram_range=(1,2),
			token_pattern=r'\b\w+\b', min_df=1)
		corpus = []
		if indices != None:
			for index in indices:
				corpus.append(self.corpus[index])
		else:
			corpus = self.corpus

		X = vectorizer.fit_transform(corpus)
		return X

	# returns a fitted vectorizer that is trained on the corpus 
	# (or a subset of the corpus specified by certain indices) that can be used to
	# convert test string corpuses into count matrices with vectorizer.transform(testcorpus)
	def extractNBCountMatrixFittedVectorizer(self, indices=None):
		vectorizer = CountVectorizer(ngram_range=(1,2),
									token_pattern=r'\b\w+\b',
									min_df=1)
		corpus = []
		if indices != None:
			for index in indices:
				corpus.append(self.corpus[index])
		else:
			corpus = self.corpus
		vectorizer.fit_transform(corpus)
		return vectorizer

	def extractNBTestCountMatrix(self, testcorpus):
		return self.vectorizer.transform(testcorpus)

	# This is used for evaluatenb classifier!
	def extractTrainingAndTestCountMatrices(self, training_indices):
		vectorizer = CountVectorizer(ngram_range=(1,2),
			token_pattern=r'\b\w+\b', min_df=1)
		training_corpus, testing_corpus = [], []
		
		for i in range(len(self.raw_test_data)):
			if (i in training_indices):
				training_corpus.append(self.corpus[i])
			else:
				testing_corpus.append(self.corpus[i])

		trainX = vectorizer.fit_transform(training_corpus)
		testX = vectorizer.transform(testing_corpus)
		return trainX, testX

	def extractLabelConfidences(self):
		confidence={'sentiment':[], 'event':[], 'time':[]}
		for example in self.raw_test_data:
			#confidence['sentiment'].append(dict([(key, example[key]) for key in example if key[0] =='s']))
			#confidence['event'].append(dict([(key, example[key]) for key in example if key[0] =='k']))
			#confidence['time'].append(dict([(key, example[key]) for key in example if key[0] =='w' ]))
			confidence['sentiment'].append(self.getConfidenceVector(example, 'sentiment'))
			confidence['event'].append(self.getConfidenceVector(example, 'event'))
			confidence['time'].append(self.getConfidenceVector(example, 'time'))
		return confidence

	def extractFullLabelConfidenceVectors(self):
		fullLabelConfidenceVectors = []
		for example in self.raw_test_data:
			confidenceVector = []
			#confidence['sentiment'].append(dict([(key, example[key]) for key in example if key[0] =='s']))
			#confidence['event'].append(dict([(key, example[key]) for key in example if key[0] =='k']))
			#confidence['time'].append(dict([(key, example[key]) for key in example if key[0] =='w' ]))
			confidenceVector += self.getConfidenceVector(example, 'sentiment')
			confidenceVector += self.getConfidenceVector(example, 'time')
			confidenceVector += self.getConfidenceVector(example, 'event')
			fullLabelConfidenceVectors.append(confidenceVector)
		return fullLabelConfidenceVectors

	# convert the dictionary representing an example into a list of confidences for 
	# a single label type
	def getConfidenceVector(self, example, labeltype):
		confidencevector = [] 
		for labelname in self.labelnames[labeltype]:
			confidencevector.append(float(example[labelname]))
			#print labelname
		return confidencevector
	
	#extracts a dictionary containing bit vectors for each label for each training
	#instance: 1 if greater than threshold and 0 otherwise
	# def getBitVector(self, examplevector, threshold):
	# 	bitvector = [0 for _ in examplevector]
	# 	for i in range(len(examplevector)):
	# 		x = examplevector[i]
	# 		if x > threshold:
	# 			bitvector[i] = 1
	# 	return bitvector
		
	# def getBitString(self, examplevector, threshold):
	# 	bitvector = [0 for _ in examplevector]
	# 	for i in range(len(examplevector)):
	# 		x = examplevector[i]
	# 		if x > threshold:
	# 			bitvector[i] = 1
	# 	return str(bitvector)

	def getMaxLabel(self, examplevector):
		maxconfidence = max(examplevector)
		maxlabel = examplevector.index(maxconfidence)
		return maxlabel


	def extractLabelBitVectors(self, threshold, indices=None):
		bitvectors = {'sentiment':[], 'time':[], 'event':[]}
		for example_idx in range(len(self.raw_test_data)):
			if indices != None and example_idx not in indices: continue
			example = self.raw_test_data[example_idx]
			bitvectors['sentiment'].append((self.getSentimentBitVector(example)))
			bitvectors['time'].append((self.getTimeBitVector(example)))
			bitvectors['event'].append((self.getEventBitVector(example, event_conf_threshold)))
		return bitvectors

	# combines the bitvectors or bitstrings in a dict to a long vector/string
	def combineLabelBitVectors(self, bitvectorDict):
		numVectors = len(bitvectorDict['sentiment'])
		full_vectors = []
		for i in range(numVectors):
			newVector = []
			for labeltype in self.labeltypes:
				newVector += bitvectorDict[labeltype][i]
			full_vectors.append(newVector)
		return full_vectors

	def extractLabelBitStrings(self, event_conf_threshold, indices=None):
		bitvectors = {'sentiment':[], 'time':[], 'event':[]}
		for example_idx in range(len(self.raw_test_data)):
			if indices != None and example_idx not in indices: continue
			example = self.raw_test_data[example_idx]
			bitvectors['sentiment'].append(' '.join([str(item) for item in self.getSentimentBitVector(example)]))
			bitvectors['time'].append(' '.join([str(item) for item in self.getTimeBitVector(example)]))
			bitvectors['event'].append(' '.join([str(item) for item in self.getEventBitVector(example, event_conf_threshold)]))
		return bitvectors

	def bitstringToIntList(self, bitstring):
		newList = bitstring[1:len(bitstring)-1]
		newList = [int(val) for val in newList.split(',')]
		return newList

	#outputs a set of one index for each label class representing most likely
	#label, ignoring any other nonzero confidences. To make dataset more complete, might want to recopy training examples 
	#with multiple labels. 
	def extractLabelIndices(self):
		confidences = self.extractLabelConfidences()
		labeldicts = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			sentimentlabel = self.getMaxLabel(confidences['sentiment'][i]) # sentiment label with highest confidence - ignore others
			eventlabel = self.getMaxLabel(confidences['event'][i])
			timelabel = self.getMaxLabel(confidences['time'][i])
			labeldicts['sentiment'].append(sentimentlabel)
			labeldicts['event'].append(eventlabel)
			labeldicts['time'].append(timelabel)
		return labeldicts

	#outputs a set of one label for each label class representing most likely
	#label, ignoring any other nonzero confidences.
	def extractLabelNames(self):
		confidences = self.extractLabelConfidences()
		labeldicts = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			sentimentindex = self.getMaxLabel(confidences['sentiment'][i]) # sentiment label with highest confidence - ignore others
			eventindex = self.getMaxLabel(confidences['event'][i])
			timeindex = self.getMaxLabel(confidences['time'][i])
			labeldicts['sentiment'].append(self.labelnames['sentiment'][sentimentindex])
			labeldicts['event'].append(self.labelnames['event'][eventindex])
			labeldicts['time'].append(self.labelnames['time'][timeindex])
		return labeldicts


	# hacky solution to get the number of labels in each category
	def getNumLabels(self):
		# confidences = self.extractLabelConfidences()
		# sentimentbitvector = self.getBitVector(confidences['sentiment'][0], 0)
		# eventbitvector = self.getBitVector(confidences['event'][0], 0)
		# timebitvector = self.getBitVector(confidences['time'][0], 0)
		# return {'sentiment':len(sentimentbitvector), 'event':len(eventbitvector), 'time':len(timebitvector)}

		# no longer hacky
		return {'sentiment':len(self.labelnames['sentiment']), 'event':len(self.labelnames['event']), 'time':len(self.labelnames['time'])}

	# Returns all unigram words after lower-casing and removing punctuations
	def getAllWords(self):
		if self.all_words != None: return self.all_words

		words = []
		for tweet in self.corpus:
			words += [word.lower().translate(string.maketrans("",""), string.punctuation) for word in tweet.split()]

		# Save for next time
		if self.all_words == None: self.all_words = words
		return words

	# Returns all unigram words
	def getAllBigramWords(self):
		if self.bigram_words != None: return self.bigram_words

		bigram_words = []
		for tweet in self.corpus:
			prevWord = '<START>'
			words = tweet.split()
			for word in words:
				word = word.lower().translate(string.maketrans("",""), string.punctuation)
				bigram_words.append((prevWord, word))
				prevWord = word

		if self.bigram_words == None: self.bigram_words = bigram_words
		return bigram_words

	def extractFullLabelBitVectors(self, event_conf_threshold):
		fullLabelBitVectors = []
		for example in self.raw_test_data:
			bitVector = self.getSentimentBitVector(example) + \
						self.getTimeBitVector(example) + \
						self.getEventBitVector(example, event_conf_threshold)
			fullLabelBitVectors.append(bitVector)
		return fullLabelBitVectors


	def getSentimentBitVector(self, example):
		sentConfVec = self.getConfidenceVector(example, 'sentiment')
		indexOfMaxConf = sentConfVec.index(max(sentConfVec))
		sentBitVec = [0 for _ in range(len(sentConfVec))]
		sentBitVec[indexOfMaxConf] = 1
		return sentBitVec

	def getTimeBitVector(self, example):
		timeConfVec = self.getConfidenceVector(example, 'time')
		indexOfMaxConf = timeConfVec.index(max(timeConfVec))
		timeBitVec = [0 for _ in range(len(timeConfVec))]
		timeBitVec[indexOfMaxConf] = 1
		return timeBitVec

	def getEventBitVector(self, example, threshold):
		eventConfVec = self.getConfidenceVector(example, 'event')
		eventBitVec = [1 if conf >= threshold else 0 for conf in eventConfVec]
		return eventBitVec