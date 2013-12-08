### Simon Notes ###
# Note: need to make different predictions lists for each classifier 
# Note: need to get actual label vectors


import vectorToLabel
tweetsToBeTagged = load_tweets()



import combinedNaiveBayes
cnbc = combinedNaiveBayes.combinedNBClassifier(data_filename='data/train.csv', numFolds=0)
cnbc_predictions_list = cnbc.combined_classify_tweets(tweetsToBeTagged)


import structuredNaiveBayes
snbc = structuredNaiveBayes.structuredNBClassifier(data_filename='data/train.csv', numFolds=0)
snbc_predictions_list = snbc.combined_classify_tweets(tweetsToBeTagged)

converter = vectorToLabel.Converter()






for i in range(len(tweetsToBeTagged)):
	print 'For tweet: %s' %(tweetsToBeTagged[i])
	print '\tPredicted: %s' %(predictions_list[i])
	labels = converter.convertToLabels(predictions_list[i])
	for labeltype in converter.labeltypes:
		print '\tPredicted %s labels: %s' %(labeltype, labels[labeltype])

