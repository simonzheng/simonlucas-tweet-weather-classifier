from csp import CSP
from feature_extraction import dataloader
#import evaluation
from sklearn.naive_bayes import MultinomialNB
# Train Classifiers for each label type

loader = dataloader.DataLoader('data/train.csv')
trainX = loader.extractNBCountMatrix()
trainlabelindices = loader.extractLabelIndices()
#evaluator = evaluation.Evaluator()
classifiers = {}
testx = trainX[0]# 
for labeltype in ['sentiment', 'event', 'time']:
	nbclassifier = MultinomialNB()
	y = trainlabelindices[labeltype]
	#nbclassifier.fit(trainX, y)
	classifiers[labeltype] = nbclassifier
	#print nbclassifier.predict_proba(testx)

# create a CSP with a variable for each possible label in each label class with a 
# domain of 0, 1 with a unary potential that corresponds to exp(P(y|x)) (this is the posterior 
# where the total probability is associated with just the other labels in the class- will this cause
# problems?
testCSP = CSP()

# add a variable and unary potential for each label 
numlabelsdict = loader.getNumLabels()
print numlabelsdict
#for labeltype in ['sentiment', 'event', 'time']:
	#numlabels = 
# 	labelunaries = classifier[labeltype].predict_proba(testx)
# 	for index in trainlabelindices[labeltype]:
# 		testcsp.add_variable(labeltype + str(index))

	
