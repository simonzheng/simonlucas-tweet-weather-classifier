#Stupid_Classifier is a bianry classifier initialized with a list of words that are 
#relevant to the positive label. If there are any matches with the good words in the 
#test set then the classifier will return 1. Else it will return 0. 
class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.labels = labels
        self.classifiers = classifiers
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
       

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return max(self.classify(x), key=lambda x: x[1])[0]
        # END_YOUR_CODE

class OneVsAllClassifier():
    def __init__(self, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        self.classifiers = classifiers


    def classify(self, x):
        """
        @param string x: the feature vector
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        output = [] 
        for c in self.classifiers:
            output.append(c.classify(x))
        return output
        # END_YOUR_CODE

class Stupid_Classifier():
	def __init__(self, goodwords):
		self.goodwords = goodwords
	#Classify looks at a word counter and goodwords
	def classify(self, frequencies):
		for w1 in frequencies:
			for w2 in self.goodwords:
				if w2 in w1: 
					return 1
		return 0
#Stupid factory returns a onevall classifier for a set of goodwords 
class StupidFactory():
    def getClassifier(self,wordlists):
        stupidclassifiers = []
        for wordlist in wordlists:
            stupidclassifier = Stupid_Classifier(wordlist)
            stupidclassifiers.append(stupidclassifier)
        onevall = OneVsAllClassifier(stupidclassifiers)
        return onevall


