import csv
import collections
from sklearn.feature_extraction.text import CountVectorizer
# DataLoader loads a dictionary representing the raw csv data 
class DataLoader:
	def __init__(self, testfile):
		self.raw_test_data = self.loadTrainData(testfile)
	def loadTrainData(self, filename):
		raw_data = []
		with open(filename, 'rb') as csvfile:
			reader = csv.DictReader(csvfile, delimiter=',')
			for row in reader:
				raw_data.append(row)
		return raw_data
		
		# returns an array of counters
	def extractFeatureVectors(self):
		features = []
		for entry in self.raw_test_data:
			text = entry["tweet"]
			features.append(collections.Counter(text.split(' ')))
		return features
	# returns a matrix representing word count that can be used to train a scikit classifier
	def extractNBCountMatrix(self):
		vectorizer = CountVectorizer(min_df=1)
		corpus = [entry['tweet'] for entry in self.raw_test_data]
		X = vectorizer.fit_transform(corpus)
		return X

	def extractLabelConfidences(self):
		confidence={'sentiment':[], 'event':[], 'time':[]}
		for example in self.raw_test_data:
			#confidence['sentiment'].append(dict([(key, example[key]) for key in example if key[0] =='s']))
			#confidence['event'].append(dict([(key, example[key]) for key in example if key[0] =='k']))
			#confidence['time'].append(dict([(key, example[key]) for key in example if key[0] =='w' ]))
			confidence['sentiment'].append(self.getConfidenceVector(example, 's'))
			confidence['event'].append(self.getConfidenceVector(example, 'k'))
			confidence['time'].append(self.getConfidenceVector(example, 'w'))
		return confidence
	# convert the dictionary representing an example into a list of confidences for 
	# a single label type
	def getConfidenceVector(self, exampledict, prefix ):
		confidencevector = []
		for key in exampledict:
			if key[0] == prefix and len(key) < 5: #HACK differentiate key "state" and "s1"
				confidencevector.append(float(exampledict[key]))
		return confidencevector
	
	#extracts a dictionary containing bit vectors for each label for each training
	#instance - 1 if greater than threshold and 0 otherwise
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


	def extractLabelBitVectors(self, threshold):
		confidences = self.extractLabelConfidences()
		bitvectors = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			#sentimentbitvector = dict([(key,1) if confidences['sentiment'][i][key] > threshold else (key,0) for key in confidences['sentiment'][i] ])
			sentimentbitvector = self.getBitVector(confidences['sentiment'][i], threshold)
			eventbitvector = self.getBitVector(confidences['event'][i], threshold)
			timebitvector = self.getBitVector(confidences['time'][i], threshold)
			bitvectors['sentiment'].append(sentimentbitvector)
			bitvectors['event'].append(eventbitvector) 
 			bitvectors['time'].append(timebitvector) 
		return bitvectors
	def extractLabelBitStrings(self, threshold):
		confidences = self.extractLabelConfidences()
		bitvectors = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			#sentimentbitvector = dict([(key,1) if confidences['sentiment'][i][key] > threshold else (key,0) for key in confidences['sentiment'][i] ])
			sentimentbitvector = self.getBitString(confidences['sentiment'][i], threshold)
			eventbitvector = self.getBitString(confidences['event'][i], threshold)
			timebitvector = self.getBitString(confidences['time'][i], threshold)
			bitvectors['sentiment'].append(sentimentbitvector)
			bitvectors['event'].append(eventbitvector) 
 			bitvectors['time'].append(timebitvector) 
		return bitvectors
	#outputs a set of one index for each label class representing most likely
	#label 
	def extractLabelIndices(self):
		confidences = self.extractLabelConfidences()
		labeldicts = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			sentimentlabel = self.getMaxLabel(confidences['sentiment'][i])
			eventlabel = self.getMaxLabel(confidences['event'][i])
			timelabel = self.getMaxLabel(confidences['time'][i])
			labeldicts['sentiment'].append(sentimentlabel)
			labeldicts['event'].append(eventlabel)
			labeldicts['time'].append(timelabel)
		return labeldicts

# hacky solution to get the number of labels in each category
	def getNumLabels(self):
		confidences = self.extractLabelConfidences()
		sentimentbitvector = self.getBitVector(confidences['sentiment'][0], 0)
		eventbitvector = self.getBitVector(confidences['event'][0], 0)
		timebitvector = self.getBitVector(confidences['time'][0], 0)
		return {'sentiment':len(sentimentbitvector), 'event':len(eventbitvector), 'time':len(timebitvector)}



