import csv
import collections
from sklearn.feature_extraction.text import CountVectorizer
# DataLoader loads a dictionary representing the raw csv data 
class DataLoader:
	def __init__(self, testfile):
		self.loadRawData(testfile)
		self.labelnames = {'sentiment':['s1', 's2', 's3', 's4', 's5'], 
		'time':['w1', 'w2', 'w3', 'w4'],
		'event':['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']
		}
		self.vectorizer = CountVectorizer(min_df=1)
		self.corpus = [entry['tweet'] for entry in self.raw_data]


	
	#turns raw input into an array of example dictionaries 
	def loadRawData(self, filename):
		raw_data = []
		with open(filename, 'rb') as csvfile:
			reader = csv.DictReader(csvfile, delimiter=',')
			for row in reader:
				raw_data.append(row)
			#self
		self.raw_data = raw_data

		# returns an array of counters that represents the occurence of 
		# each word - used in stupidclassifier 
	def extractFeatureVectors(self):
		features = []
		for entry in self.raw_data:
			text = entry["tweet"]
			features.append(collections.Counter(text.split(' ')))
		return features
	# returns a matrix representing word count that can be used to train a scikit classifier
	# - used in NB classifier
	def extractNBCountMatrix(self):
		X = self.vectorizer.fit_transform(self.corpus)
		return X

	# Takes in a test corpus and transforms it into a NBCountMatrix using the vectorizer
	# that was fit to the training corpus
	def extractNBTestCountMatrix(self, testcorpus):
		return self.vectorizer.transform(testcorpus)

	# extract the confidences for each label class from each example
	def extractLabelConfidences(self):
		confidence={'sentiment':[], 'event':[], 'time':[]}
		for example in self.raw_data:
			#confidence['sentiment'].append(dict([(key, example[key]) for key in example if key[0] =='s']))
			#confidence['event'].append(dict([(key, example[key]) for key in example if key[0] =='k']))
			#confidence['time'].append(dict([(key, example[key]) for key in example if key[0] =='w' ]))
			confidence['sentiment'].append(self.getConfidenceVector(example, 'sentiment'))
			confidence['event'].append(self.getConfidenceVector(example, 'event'))
			confidence['time'].append(self.getConfidenceVector(example, 'time'))
		return confidence

	# extract the confidences from a given label type (denoted by prefix) from 
	# a single example
	def getConfidenceVector(self, example, sentiment):
		confidencevector = [] 
		for labelname in self.labelnames[sentiment]:
			confidencevector.append(float(example[labelname]))
			#print labelname
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

	#string is hashable - works with NBClassifier
	def getBitString(self, examplevector, threshold):
		bitvector = [] 
		for x in examplevector:
			if x > threshold:
				bitvector.append(1)
			else:
				bitvector.append(0)
		return str(bitvector)

	#get the highest confidence label for a given confidence vector 
	def getMaxLabel(self, examplevector):
		maxconfidence = max(examplevector)
		maxlabel = examplevector.index(maxconfidence)
		return maxlabel

	# returns an array of bit vector arrays containing all the label information for that example in the form 
	# ['s1', 's2', 's3', 's4', 's5','k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15','w1', 'w2', 'w3', 'w4']
	def extractCompositeLabelBitVectors(self, threshold):
		print threshold
		bitvector = []
		for example in self.raw_data:
			examplebitvector = []
			for labeltype in ['sentiment', 'event', 'time']:
				labelconfidencevector = [float(example[labelname]) for labelname in self.labelnames[labeltype]]
				#print labelconfidencevector
				labelbitvector = self.getBitVector(labelconfidencevector, threshold)
				#print labelbitvector
				examplebitvector += labelbitvector
			bitvector.append(examplebitvector)
		return bitvector

	# returns dictionary { sentiment:[] event:[] time:[] } with an array for each 
 	# example representing the binary bit vector with confidence thresholded by the
 	# threshold parameter
	def extractLabelBitVectors(self, threshold):
		confidences = self.extractLabelConfidences()
		bitvectors = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_data)):
			#sentimentbitvector = dict([(key,1) if confidences['sentiment'][i][key] > threshold else (key,0) for key in confidences['sentiment'][i] ])
			sentimentbitvector = self.getBitVector(confidences['sentiment'][i], threshold)
			eventbitvector = self.getBitVector(confidences['event'][i], threshold)
			timebitvector = self.getBitVector(confidences['time'][i], threshold)
			bitvectors['sentiment'].append(sentimentbitvector)
			bitvectors['event'].append(eventbitvector) 
 			bitvectors['time'].append(timebitvector) 
		return bitvectors

	# returns dictionary { sentiment:[] event:[] time:[] } with a string for each 
 	# example representing the binary bit vector with confidence thresholded by the
 	# threshold parameter
	def extractLabelBitStrings(self, threshold):
		confidences = self.extractLabelConfidences()
		bitvectors = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_data)):
			#sentimentbitvector = dict([(key,1) if confidences['sentiment'][i][key] > threshold else (key,0) for key in confidences['sentiment'][i] ])
			sentimentbitvector = self.getBitString(confidences['sentiment'][i], threshold)
			eventbitvector = self.getBitString(confidences['event'][i], threshold)
			timebitvector = self.getBitString(confidences['time'][i], threshold)
			bitvectors['sentiment'].append(sentimentbitvector)
			bitvectors['event'].append(eventbitvector) 
 			bitvectors['time'].append(timebitvector) 
		return bitvectors

	#outputs a set of one index for each label class representing most likely
	#label, ignoring any other nonzero confidences. To make dataset more complete, might want to recopy training examples 
	#with multiple labels. 
	def extractLabelIndices(self):
		confidences = self.extractLabelConfidences()
		labeldicts = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_data)):
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
		for i in range(len(self.raw_data)):
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



