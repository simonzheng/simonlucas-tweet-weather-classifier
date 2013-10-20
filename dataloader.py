import csv



def LoadTrainData(filename):
	tweets = []
	labels = [] 
	sentimentlabels = [] 
	timelabels = [] 
	weatherlabels = []

	raw_data = []
	with open(filename, 'rb') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\n')
		for row in reader: 
			raw_data.append(row)
	print raw_data

LoadTrainData('train.csv')