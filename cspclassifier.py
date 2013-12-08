#from newCombinedNaiveBayes import combinedNBClassifier
from feature_extraction import dataloader
from backtrackingsearch import BacktrackingSearch
from constraint import *
from csp import CSP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import vectorToLabel
import time
import evaluation
import math
from collections import Counter
import pickle
import csv

class CSPClassifier:
	def __init__(self, data_filename):
		self.loader = dataloader.DataLoader(data_filename)
		self.sentiment_label_indices = range(0,5)
		self.time_label_indices = range(5,9)
		self.event_label_indices = range(9, 24)
		self.label_types = ['sentiment', 'time', 'event']
		self.gold_bitvectors = self.loader.extractFullLabelBitVectors(1.0/3)
		self.label_types = ['sentiment', 'time', 'event']
		self.totalNumLabels = self.loader.totalNumLabels
		self.numDataPoints = self.loader.numDataPoints
		self.binaryconstraints = self.loadConstraints('binary-constraints.csv')
		self.evaluator = evaluation.Evaluator()

	def loadConstraints(self, filename):
   		with open(filename, 'r') as f:
			data = [row for row in csv.reader(f.read().splitlines())]
		return data

	def evaluate(self, numFolds):
		kf = KFold(self.loader.numDataPoints, n_folds=numFolds, indices=True)
		all_rmse, all_rmse_by_class = [], []
		all_abs_acc, all_abs_acc_by_class, all_predicted = [], [], []
		current_fold = 1
		for train_indices, test_indices in kf:
			print current_fold
			#start_time = time.time()
			# get training tweets and gold
			self.training_tweets = [self.loader.corpus[train_idx] for train_idx in train_indices]
			self.training_gold = [self.gold_bitvectors[train_idx] for train_idx in train_indices]
				
			# fit vectorizer
			self.vectorizer = self.loader.extractNBCountMatrixFittedVectorizer()
			self.featurized_training_tweets = self.vectorizer.fit_transform(self.training_tweets)
			# print 'finished fitting vectorizer and fitting training_tweets after this many seconds', time.time() - start_time
			# print self.vectorizer.get_feature_names()

			# get test tweets and gold
			self.test_tweets = [self.loader.corpus[test_idx] for test_idx in test_indices]
			self.test_gold = [self.gold_bitvectors[test_idx] for test_idx in test_indices]
			self.classifiers = self.getClassifiers()

			#create csp for test tweets
			fold_predicted, rmse, rmse_by_class, absolute_accuracy, absolute_accuracy_by_class = self.cspClassify(self.test_tweets, self.test_gold, True)
			all_rmse.append(rmse)
			all_rmse_by_class.append(rmse_by_class)
			all_abs_acc.append(absolute_accuracy)
			all_abs_acc_by_class.append(absolute_accuracy_by_class)
			current_fold +=1 
			all_predicted = all_predicted + fold_predicted
		print 'overall rmse = ', np.mean(all_rmse)
		print 'overall rmse by class = '
		for label_type in self.label_types:
			print '\t', label_type, np.mean([fold_rmse[label_type] for fold_rmse in all_rmse_by_class])
		print 'overall absolute_accuracy = ', np.mean(all_abs_acc)
		print 'overall absolute_accuracy_by_class = ' 
		for label_type in self.label_types:
			print '\t', label_type, np.mean([fold_rmse[label_type] for fold_rmse in all_abs_acc_by_class])
		print 'dumping predicted vectors to pickle file'
		pickle.dump(all_predicted, open('csppredicted.pkl', "wb"))
			
	def add_binary_constraints(self, csp):
		for constraint in self.binaryconstraints:
 			label1 = constraint[0]
 			label2 = constraint[1]
 			# We add a constraint for each pair of labels in our constraint specifying that we cannot have both of them co-occurring (i.e. turned to 1)
 			if label1 in csp.varNames and label2 in csp.varNames:
 				csp.add_binary_potential(label1, label2, lambda l1, l2 : l1 + l2 <= 1)
	def cspClassify(self, tweets, test_gold, constraints = False):
		gold_bitvectors = test_gold
		predictedvectors = []
		probabilitythreshold = .1
		print 'classifying tweets'
		numTweets = len(tweets)
		featurized_tweets = self.vectorizer.transform(tweets)
		probabilities = [self.classifiers[idx].predict_proba(featurized_tweets) for idx in range(self.totalNumLabels)]
		for exampleindex in range(len(tweets)):
			#start_time = time.time()
			example = featurized_tweets[exampleindex]
			backsearch = BacktrackingSearch() 
			csp = CSP()
			for label in range(self.totalNumLabels):
				#start_time = time.time()
				probability = probabilities[label][exampleindex][1]
				# end_time = time.time()
				# elapsed_time = end_time - start_time

				# print 'get probability took: ', elapsed_time


				varname = self.loader.ordered_keys[label]
				if probability > probabilitythreshold:
					csp.add_variable(varname, [0,1])
					csp.add_unary_potential(varname, lambda x: math.pow(probability, x) * math.pow(1-probability, 1-x))	

			if constraints:
				self.add_binary_constraints(csp)
			#end_time = time.time()
			#elapsed_time = end_time - start_time
			#print 'csp formation took ', elapsed_time
			


			start_time = time.clock()
			backsearch.solve(csp, True, True, True)
			optimalAssignment = Counter(backsearch.optimalAssignment)
			end_time = time.clock()
			elapsed_time = end_time - start_time
			#print 'backtracking took: ', elapsed_time
			predictedvector = []
			for label in range(self.totalNumLabels):
				predictedvector.append(optimalAssignment[self.loader.ordered_keys[label]])
			predictedvectors.append(predictedvector)

		rmse = self.evaluator.rmse(predictedvectors, gold_bitvectors)
		print 'rmse:', rmse
		rmse_by_class = self.evaluator.rmse_by_labelclass(predictedvectors, gold_bitvectors)
		print 'rmse_by_class', rmse_by_class

		absolute_accuracy = self.evaluator.absolute_accuracy(predictedvectors, gold_bitvectors)
		print 'absolute_accuracy', absolute_accuracy

		absolute_accuracy_by_class = self.evaluator.absolute_accuracy_by_labelclass(predictedvectors, gold_bitvectors)
		print 'absolute_accuracy_by_class', absolute_accuracy_by_class

		return predictedvectors, rmse, rmse_by_class, absolute_accuracy, absolute_accuracy_by_class
	def getClassifiers(self):
		classifiers = []
		for label_index in range(self.totalNumLabels):
			x = self.featurized_training_tweets
			y = [gold_vec[label_index] for gold_vec in self.training_gold]
			classifier = MultinomialNB().fit(x, y)
			classifiers.append(classifier)
		print 'returning %i classifiers' %(len(classifiers))
		return classifiers

