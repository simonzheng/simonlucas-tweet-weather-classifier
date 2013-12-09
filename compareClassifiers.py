### Simon Notes ###
# Note: need to make different predictions lists for each classifier 
# Note: need to get actual label vectors
from feature_extraction import dataloader

import vectorToLabel
import pickle
# tweetsToBeTagged = load_tweets()
loader =dataloader.DataLoader('data/train.csv')
totalNumLabels = loader.totalNumLabels
numDataPoints = loader.numDataPoints
tweetcorpus = loader.corpus
converter = vectorToLabel.Converter()



# import combinedNaiveBayes
# cnbc = combinedNaiveBayes.combinedNBClassifier(data_filename='data/train.csv', numFolds=0)
# cnbc_predictions_list = cnbc.combined_classify_tweets(tweetsToBeTagged)


# import structuredNaiveBayes
# snbc = structuredNaiveBayes.structuredNBClassifier(data_filename='data/train.csv', numFolds=0)
# snbc_predictions_list = snbc.combined_classify_tweets(tweetsToBeTagged)



def compare_combinedNB_csp(predictions, goldvectors):
	for exampleindex in range(numDataPoints):
		#if predictions['csp'][exampleindex] != predictions['structurednb']:
		tweet = tweetcorpus[exampleindex]
		truevalue = goldvectors[exampleindex]
		cspprediction = predictions['csp'][exampleindex]
		combinednbprediction = predictions['combinednb'][exampleindex]
		if cspprediction != combinednbprediction:
			cspcorrect = cspprediction == truevalue
			combinednbcorrect = combinednbprediction == truevalue
			print 'tweet:'
			print tweet
			print 'correct:'
			if cspcorrect and combinednbcorrect:
				print 'BOTH'
			elif cspcorrect:
				print 'CSP'
			elif combinednbcorrect:
				print 'COMBINEDNB'
			else:
				print 'NEITHER'
			print 'true value'
			print converter.convertToLabels(truevalue)
			print 'csp prediction'
			print converter.convertToLabels(cspprediction)
			print 'combined nb prediction'
			print converter.convertToLabels(combinednbprediction)
def compare_structuredNB_combinedNB(predictions, goldvectors):
	for exampleindex in range(numDataPoints):
		#if predictions['csp'][exampleindex] != predictions['structurednb']:
		tweet = tweetcorpus[exampleindex]
		truevalue = goldvectors[exampleindex]
		structurednbprediction = predictions['structurednb'][exampleindex]
		combinednbprediction = predictions['combinednb'][exampleindex]
		structuredlabels = converter.convertToLabels(structurednbprediction)
		combinedlabels = converter.convertToLabels(combinednbprediction)
		truelabels = converter.convertToLabels(truevalue)
		if structuredlabels['event'] != combinedlabels['event']:
			structurednbcorrect = structurednbprediction == truevalue
			combinednbcorrect = combinednbprediction == truevalue
			print 'tweet:'
			print tweet
			print 'correct:'
			if structurednbcorrect and combinednbcorrect:
				print 'BOTH'
			elif structurednbcorrect:
				print 'STRUCTUREDNB'
			elif combinednbcorrect:
				print 'COMBINEDNB'
			else:
				print 'NEITHER'
			print 'true value'
			print converter.convertToLabels(truevalue)
			print 'structured nb prediction'
			print converter.convertToLabels(structurednbprediction)
			print 'combined nb prediction'
			print converter.convertToLabels(combinednbprediction)


predictions = {'csp': pickle.load(open('csppredicted.pkl')) , 'combinednb':pickle.load(open('combinedNBpredicted.pkl')), 'structurednb':pickle.load(open('structuredNBpredicted.pkl'))}
goldvectors = loader.extractFullLabelBitVectors()
compare_structuredNB_combinedNB(predictions, goldvectors)







# for i in range(len(tweetsToBeTagged)):
# 	print 'For tweet: %s' %(tweetsToBeTagged[i])
# 	print '\tPredicted: %s' %(predictions_list[i])
# 	labels = converter.convertToLabels(predictions_list[i])
# 	for labeltype in converter.labeltypes:
# 		print '\tPredicted %s labels: %s' %(labeltype, labels[labeltype])

