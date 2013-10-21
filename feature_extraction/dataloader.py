import csv

# DataLoader loads a dictionary representing the raw csv data 
class DataLoader:
	def _init_(testfile):
		self.raw_test_data = self.LoadTrainData(testfile)
	def loadTrainData(filename):
		raw_data = []
		with open(filename, 'rb') as csvfile:
			reader = csv.DictReader(csvfile, delimiter='\n')
			for row in reader: 
				raw_data.append(row)
		return raw_data
		
		# returns an array of counters
	def extractFeatureVectors():
		features = []
		for entry in self.raw_test_data:
			text = entry['tweet']
			features.append(Counter(text.split(' ')))
		return features

	def extractLabels():
		labeldict={'sentiment':[], 'event':[], 'time':[]}
		#for entry in self.raw.test_data:




LoadTrainData('train.csv')