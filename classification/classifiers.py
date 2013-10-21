#Stupid_Classifier is a binary classifier initialized with a list of words that are 
#relevant to the positive label. If there are any matches with the good words in the 
#test set then the classifier will return 1. Else it will return 0. 

class Stupid_Classifier(Classifier)
	def _init_(goodwords):
		self.goodwords = goodwords
	#Classify looks at a word counter and goodwords
	def classify(frequencies):
		for w1 in frequencies:
			for w2 in self.goodwords:
				if w1 == w2: 
					return 1
		return 0


