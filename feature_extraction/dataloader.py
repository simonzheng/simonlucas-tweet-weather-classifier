import csv
import collections
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

	def getConfidenceVector(self, exampledict, prefix ):
		confidencevector = []
		for key in exampledict:
			if key[0] == prefix and len(key) < 5: #differentiate key "state" and "s1"
				confidencevector.append(float(exampledict[key]))
		return confidencevector

	def getBitVector(self, examplevector, threshold):
		bitvector = [] 
		for x in examplevector:
			if x > threshold:
				bitvector.append(1)
			else:
				bitvector.append(0)
		return bitvector
		
	def extractLabelBitVectors(self, threshold):
		confidences = self.extractLabelConfidences()
		bitvectors = {'sentiment':[], 'event':[], 'time':[]}
		for i in range(len(self.raw_test_data)):
			#sentimentbitvector = dict([(key,1) if confidences['sentiment'][i][key] > threshold else (key,0) for key in confidences['sentiment'][i] ])
			sentimentbitvector = self.getBitVector(confidences['sentiment'][i], threshold)
			eventbitvector = self.getBitVector(confidences['event'][i], threshold)
			timebitvector = self.getBitVector(confidences['time'][i], threshold)
			bitvectors['sentiment'].append(sentimentbitvector)
			bitvectors['event'].append(sentimentbitvector) 
 			bitvectors['time'].append(sentimentbitvector) 
		return bitvectors


