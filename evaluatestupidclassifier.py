from classification import classifiers
from feature_extraction import dataloader
#import evaluation
import collections
goodwords = {'sentiment':[[], ['bad', 'sucks', 'awful', 'negative'], [], ['good', 'great', 'like', 'love', 'awesome'], []], 
			'time':[['is', 'are', 'now','present'], ['will', 'might', 'should','could', 'can', 'future'], [], ['did', 'had', 'went', 'happened','came', 'past', 'passed', 'was']], 
			'event':[['cloud'], ['cold'], ['dry'], ['hot'], ['humid'], ['hurricane'], [], ['ice'], [], ['rain'], ['snow'], ['storm'], ['sun'], ['tornado'], ['wind']]
}

loader = dataloader.DataLoader('data/train.csv')
testfeaturevectors = loader.extractFeatureVectors()
testlabelvectors = loader.extractLabelBitVectors()
factory = classifiers.StupidFactory()
for labeltype in goodwords:
	onevallstupid = Stupid_Factory(goodwords[labeltype])
	Get_Error(onevallstupid,testfeaturevectors, testlabelvectors)

def Get_Error(classifier, testfeaturevectors, expected):
	


