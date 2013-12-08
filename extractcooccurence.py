from collections import Counter
from itertools import combinations
from feature_extraction import dataloader
import cPickle as pickle
#load training data
loader = dataloader.DataLoader('data/train.csv')
examples = loader.extractFullLabelBitVectors(.333)
# initialize joint occurence, single occurence counters
# for each data point,  increment joint occurence for each combination of labels
# for each data point, increment total occurence for each label 
# for each combination of labels x, y cooccurence(x,y) = jointoccurence/totaloccurence(x) + totaloccurence(y)_

# labels = \
# 			{'sentiment':['s1', 's2', 's3', 's4', 's5'], 
# 			'time':['w1', 'w2', 'w3', 'w4'],
# 			'event':['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']
# 			}
# jointlabels = {'sentiment':list(combinations(sentimentlabels, 2), 
# 		'time':list(combinations(timelabels, 2),
# 		'event':list(combinations(eventlabels, 2)
# 	}
# jointoccurences = {'sentiment':Counter(), 
# 			'time':Counter(),
# 			'event': Counter() }
# occurences = {'sentiment':Counter(), 
# 			'time':Counter(),
# 			'event': Counter()
# 				}


labelnames  = ['s1', 's2', 's3', 's4', 's5', 
							'w1', 'w2', 'w3', 'w4', 
							'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 
							'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15']

jointlabels = list(combinations(labelnames, 2))
jointoccurences = Counter()
occurences = Counter()

for example in examples:
	labels = [loader.ordered_keys[i] for i in range(len(example)) if example[i] ==1]
	for jointlabel in jointlabels:
		if jointlabel[0] in labels and jointlabel[1] in labels:
			jointoccurences[jointlabel] += 1
	for label in labelnames:
		if label in labelnames:
			occurences[label] +=1 
cooccurence = {}
for jointlabel in jointlabels:
	cooccurence[jointlabel] = float(jointoccurences[jointlabel])/float(occurences[jointlabel[0]] + occurences[jointlabel[1]] - jointoccurences[jointlabel])
print cooccurence