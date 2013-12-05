# import vectorToLabel.py
# converter = vectorToLabel.Converter()

# for i in range(len(inputTestTweets)):
# 	print 'For tweet: %s' %(tweetsToBeTagged[i])
# 	print '\tPredicted: %s' %(predictions_list[i])
# 	for labeltype in labeltypes:
# 		labels = converter.convertToLabels(predictions_list[i])
# 		print '\tPredicted %s labels: %s' %(labeltype, labels[labeltype])

class Converter:
	def __init__(self):
		self.ordered_keys = ['s1', 's2', 's3', 's4', 's5', 
							'w1', 'w2', 'w3', 'w4', 
							'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 
							'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']
		self.map = {
			's1' : "I can't tell",
			's2' : "Negative",
			's3' : "Neutral / author is just sharing information",
			's4' : "Positive",
			's5' : "Tweet not related to weather condition",
			'w1' : "current (same day) weather",
			'w2' : "future (forecast)",
			'w3' : "I can't tell",
			'w4' : "past weather",
			'k1' : "clouds",
			'k2' : "cold",
			'k3' : "dry",
			'k4' : "hot",
			'k5' : "humid",
			'k6' : "hurricane",
			'k7' : "I can't tell",
			'k8' : "ice",
			'k9' : "other",
			'k10' : "rain",
			'k11' : "snow",
			'k12' : "storms",
			'k13' : "sun",
			'k14' : "tornado",
			'k15' : "wind"
		}
		self.numKeys = len(self.map)

	def convertToLabels(self, bitvector):
		labels = {	'sentiment': [],
					'event' : [],
					'time' : []
					}
		keys = self.ordered_keys
		for i in range(self.numKeys):
			if bitvector[i] == 1: 
				if i >= 0 and i <=4:	# append sentiment label
					labels['sentiment'].append(self.map[keys[i]])
				elif i >= 5 and i <= 8:	# append time label
					labels['time'].append(self.map[keys[i]])
				else:
					labels['event'].append(self.map[keys[i]])
		return labels