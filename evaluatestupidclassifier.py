from classification import classifiers
from feature_extraction import dataloader
import evaluation
#import evaluation
import collections
goodwords = {'sentiment':[[], ['bad', 'sucks', 'awful', 'negative'], [], ['good', 'great', 'like', 'love', 'awesome'], []], 
			'time':[['is', 'are', 'now','present'], ['will', 'might', 'should','could', 'can', 'future'], [], ['did', 'had', 'went', 'happened','came', 'past', 'passed', 'was']], 
			'event':[['cloud'], ['cold'], ['dry'], ['hot'], ['humid'], ['hurricane'], [], ['ice'], [], ['rain'], ['snow'], ['storm'], ['sun'], ['tornado'], ['wind']]
}

loader = dataloader.DataLoader('data/train.csv')
testfeaturevectors = loader.extractFeatureVectors()
testlabelvectors = loader.extractLabelBitVectors(.5)
evaluator = evaluation.Evaluator()
factory = classifiers.StupidFactory()
for labeltype in goodwords:
	onevallstupid = factory.getClassifier(goodwords[labeltype])
	expectedvectors = testlabelvectors[labeltype]
	predictedvectors = []
	for example in testfeaturevectors:
		predicted = onevallstupid.classify(example)
		predictedvectors.append(predicted)
	#print labeltype
	#print 'numrows' ,len(predictedvectors)
	#print 'numrows' ,len(expectedvectors)
	#print 'predicted numcols', len(predictedvectors[0]),predictedvectors[0]
	#print 'expected numcols', len(expectedvectors[0]), expectedvectors[0]
	rmse = evaluator.rmse(predictedvectors, expectedvectors)
	error_rate = evaluator.error_rate(predictedvectors, expectedvectors)
	print 'rmse', labeltype, ' : ', rmse
	print 'error_rate', labeltype, ' : ', error_rate
		


