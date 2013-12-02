import csv
import collections
from sklearn.feature_extraction.text import CountVectorizer
# DataLoader loads a dictionary representing the raw csv data 
class DataLoader:
	def __init__(self, testfile):
		self.raw_test_data = self.loadTrainData(testfile)
		self.numDataPoints = len(self.raw_test_data)
		self.labelnames = {'sentiment':['s1', 's2', 's3', 's4', 's5'], 
		'time':['w1', 'w2', 'w3', 'w4'],
		'event':['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']
		}
		self.vectorizer = CountVectorizer(min_df=1)
		self.corpus = [entry['tweet'] for entry in self.raw_test_data]
 
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
		vectorizer = CountVectorizer(min_df=1)
		corpus = []
		for i in range(len(self.raw_test_data)):
			if (indices != None and i not in indices): continue
			corpus.append(self.raw_test_data[i]['tweet'])

		X = vectorizer.fit_transform(corpus)
		return X

	def extractNBTestCountMatrix(self, testcorpus):
		return self.vectorizer.transform(testcorpus)

	# This is used for evaluatenb classifier!
	def extractTrainingAndTestCountMatrices(self, training_indices):
		vectorizer = CountVectorizer(min_df=1)
		training_corpus = []
		testing_corpus = []
		for i in range(len(self.raw_test_data)):
			if (i in training_indices):
				training_corpus.append(self.raw_test_data[i]['tweet'])
			else:
				testing_corpus.append(self.raw_test_data[i]['tweet'])
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

	# convert the dictionary representing an example into a list of confidences for 
	# a single label type
	def getConfidenceVector(self, example, sentiment):
		confidencevector = [] 
		for labelname in self.labelnames[sentiment]:
			confidencevector.append(float(example[labelname]))
			#print labelname
		return confidencevector
	
	#extracts a dictionary containing bit vectors for each label for each training
	#instance: 1 if greater than threshold and 0 otherwise
	def getBitVector(self, examplevector, threshold):
		bitvector = [] 
		for x in examplevector:
			if x > threshold:
				bitvector.append(1)
			else:
				bitvector.append(0)
		return bitvector
		
	def getBitString(self, examplevector, threshold):
		bitvector = [] 
		for x in examplevector:
			if x > threshold:
				bitvector.append(1)
			else:
				bitvector.append(0)
		return str(bitvector)

	def getMaxLabel(self, examplevector):
		maxconfidence = max(examplevector)
		maxlabel = examplevector.index(maxconfidence)
		return maxlabel


	def extractLabelBitVectors(self, threshold, indices=None):
		confidences = self.extractLabelConfidences()
		bitvectors = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			if (indices != None and i not in indices): continue # skip the rest of this loop if indices are specified and we are on an index that's not in the specified indices
			#sentimentbitvector = dict([(key,1) if confidences['sentiment'][i][key] > threshold else (key,0) for key in confidences['sentiment'][i] ])
			sentimentbitvector = self.getBitVector(confidences['sentiment'][i], threshold)
			eventbitvector = self.getBitVector(confidences['event'][i], threshold)
			timebitvector = self.getBitVector(confidences['time'][i], threshold)
			bitvectors['sentiment'].append(sentimentbitvector)
			bitvectors['event'].append(eventbitvector) 
 			bitvectors['time'].append(timebitvector) 
		return bitvectors

	def extractLabelBitStrings(self, threshold, indices=None):
		confidences = self.extractLabelConfidences()
		bitvectors = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			if (indices != None and i not in indices): continue # skip the rest of this loop if indices are specified and we are on an index that's not in the specified indices
			#sentimentbitvector = dict([(key,1) if confidences['sentiment'][i][key] > threshold else (key,0) for key in confidences['sentiment'][i] ])
			sentimentbitvector = self.getBitString(confidences['sentiment'][i], threshold)
			eventbitvector = self.getBitString(confidences['event'][i], threshold)
			timebitvector = self.getBitString(confidences['time'][i], threshold)
			bitvectors['sentiment'].append(sentimentbitvector)
			bitvectors['event'].append(eventbitvector) 
 			bitvectors['time'].append(timebitvector) 
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
		confidences = self.extractLabelConfidences()
		sentimentbitvector = self.getBitVector(confidences['sentiment'][0], 0)
		eventbitvector = self.getBitVector(confidences['event'][0], 0)
		timebitvector = self.getBitVector(confidences['time'][0], 0)
		return {'sentiment':len(sentimentbitvector), 'event':len(eventbitvector), 'time':len(timebitvector)}



