import numpy as np
from sklearn.naive_bayes import MultinomialNB
from feature_extraction import dataloader
# import ast
from sklearn.cross_validation import KFold
import evaluation
import time





labeltypes = ['sentiment', 'event', 'time']
filename = 'data/test_5.csv'

#load the raw train csv file into the loader
#TODO: load a separate train and held-out test file into separate loaders
loader = dataloader.DataLoader(filename)
# get the count matrix representing the wordcount of each word for each example
trainX = loader.extractNBCountMatrix()
# the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
confidencethreshold = .5
# get the label vectors in the form of bit strings ( one bit string for each label type for each examples)
trainlabelbitstrings = loader.extractLabelBitStrings(confidencethreshold) 

classifiers = {}
# train a multinomial naive bayes classifier for each label class 
for labeltype in labeltypes:
	nbclassifier = MultinomialNB()
	#the training label bit string for the given label class
	y = trainlabelbitstrings[labeltype]
	#training the classifier on examples and labels
	nbclassifier.fit(trainX, y)
	#putting the classifier in the dictionary
	classifiers[labeltype] = nbclassifier







print "\n************* Evaluating our Naive Bayes Classifier on Training Data *************\n"
#evaluate the training accuracy of the classifier for each label class using the internal MultinomialNB score method 
for labeltype in labeltypes:
	accuracy = classifiers[labeltype].score(trainX, trainlabelbitstrings[labeltype])
	print "Training Accuracy for " + labeltype + " labels: ", accuracy

######## K Means Cross-Validation with k=5 held-out test sets ###########
print "\n************* Evaluating our Naive Bayes Classifier on Test Data *************\n"
loader = dataloader.DataLoader(filename)
numFolds = 5
kf = KFold(loader.numDataPoints, n_folds=numFolds, indices=True)


fold_count = 1
all_accuracies, all_rmse = {}, {}
for labeltype in labeltypes:
	all_accuracies[labeltype] = []
	all_rmse[labeltype] = []
for train_indices, test_indices in kf:
	print 'Currently performing k-fold validation on fold iteration #%d' %(fold_count)
	print('Train: %s; Test: %s' %(train_indices, test_indices))
	# get the count matrix representing the wordcount of each word for each example
	print 'Getting training and test Count Matrices...'
	start_time = time.time()
	trainX, testX = loader.extractTrainingAndTestCountMatrices(training_indices=train_indices)
	elapsed_time = time.time() - start_time
	print 'Completed Getting training and test Count Matrices... took ', elapsed_time
	# the confidencethreshold parameter designates the cutoff confidence score for converting discrete confidences to binary values 
	confidencethreshold = .5
	# get the label vectors in the form of bit strings ( one bit string for each label type for each examples)
	print 'Getting trainlabelbitstrings with the specified indices...'
	start_time = time.time()
	trainlabelbitstrings = loader.extractLabelBitStrings(confidencethreshold, indices=train_indices)
	elapsed_time = time.time() - start_time
	print 'Completed getting trainlabelbitstrings with the specified indices... took ', elapsed_time

	print 'Getting testlabelbitvectors with the specified indices...'
	start_time = time.time()
	testlabelbitvectors = loader.extractLabelBitVectors(confidencethreshold, indices=test_indices)
	elapsed_time = time.time() - start_time
	print 'Completed getting testlabelbitvectors with the specified indices... took ', elapsed_time


	print 'Getting testlabelbitstrings with the specified indices...'
	start_time = time.time()
	testlabelbitstrings = loader.extractLabelBitStrings(confidencethreshold, indices=test_indices)
	elapsed_time = time.time() - start_time
	print 'Completed Getting testlabelbitstrings with the specified indices...', elapsed_time

	classifiers = {}
	# train a multinomial naive bayes classifier for each label class 
	for labeltype in labeltypes:
		nbclassifier = MultinomialNB()
		#the training label bit string for the given label class
		y = trainlabelbitstrings[labeltype]
		#training the classifier on examples and labels
		nbclassifier.fit(trainX, y)
		#putting the classifier in the dictionary
		classifiers[labeltype] = nbclassifier

	for labeltype in labeltypes:
		# Calculate MultinomialNB accuracy scores
		accuracy = classifiers[labeltype].score(testX, testlabelbitstrings[labeltype])
		
		# Calculate RMSE scores
		print 'Making predictions and converting prediction bitstrings to bitvectors for %s labels...' %(labeltype)
		start_time = time.time()
		predictions_list = []
		for testX_matrixcounts in testX:
			predicted_label = classifiers[labeltype].predict(testX_matrixcounts)[0]
			predicted_bitvector = loader.bitstringToIntList(predicted_label)
			predictions_list.append(predicted_bitvector)
		
		elapsed_time = time.time() - start_time
		print 'Completed converting prediction bitstrings to bitvectors... took ', elapsed_time
		gold_list = testlabelbitvectors[labeltype]

		# # Sanity check that conversion from bitstring to bitvector was successful
		# print 'gold_list[0] = ', gold_list[0]
		# print 'first char gold_list[0][0] = ', gold_list[0][0]

		# print 'predictions_list[0] = ', predictions_list[0]
		# print 'first char predictions_list[0][0] = ', predictions_list[0][0]
		
		evaluator = evaluation.Evaluator()
		rmse = evaluator.rmse(predictions_list, gold_list)
		all_rmse[labeltype].append(rmse)

		all_accuracies[labeltype].append(accuracy)
		print "Test Accuracy for " + labeltype + " labels: ", accuracy, 'for iteration: ', fold_count
		print "Test RMSE for " + labeltype + " labels: ", rmse, 'for iteration: ', fold_count

	fold_count += 1 # Update the fold count to next iteration

for labeltype in labeltypes:
	print "Overall Test Accuracy for " + labeltype + " labels: ", np.mean(all_accuracies[labeltype])
	print "Overall Test RMSE for " + labeltype + " labels: ", np.mean(all_rmse[labeltype])

#POSSIBLE TODO: perform rms error analysis using your module. 
# the output of classifiers[labeltype].predict(testX) is a bit string (required a hashable type)
# so if you can figure out how to convert a bit string to a tuple you can do rms 
# This might give us something interesting to write about in terms of error analysis


#Junk Code that might be resused 
	#predictedvectors = []
	#for example in testX:
	#	predicted = classifiers[labeltype].predict(testX)[0]
	#	predictedvectors.append(predicted)
	#expectedvectors = testY[labeltype][0:100]
	#error_rate = evaluator.error_rate(predictedvectors, expectedvectors)
