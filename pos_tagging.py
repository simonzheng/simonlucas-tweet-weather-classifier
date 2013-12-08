import os
import sys

if len(sys.argv) < 1:
    print "Usage: pos_tagging.py tagged_filename"
    print
    print "Implement PoS labeling. File format is one sentence/tweet per line, space tokenised, with tags following tokens separated by underscores"
    print " e.g. 'The_DT big_JJ dog_NN #scary_HT'"
    print "(requires NLTK 2 or above to be installed; possible via e.g. 'apt-get install python-nltk' or 'easy_install nltk')"
    sys.exit()

import nltk

# gs_corpus_path = os.path.realpath(sys.argv[1])
candidate_corpus_path = os.path.realpath(sys.argv[1])

# gs = nltk.corpus.reader.TaggedCorpusReader(os.path.dirname(gs_corpus_path), os.path.basename(gs_corpus_path), sep='_')
candidate = nltk.corpus.reader.TaggedCorpusReader(os.path.dirname(candidate_corpus_path), os.path.basename(candidate_corpus_path), sep='_')

candidate_tagged_sents = candidate.tagged_sents()

for sentence in candidate_tagged_sents:
	print '----------------------------------------------------------------------'
	print '--> The sentence is: ', sentence
	verbFound = False
	verbs = []
	for (word, tag) in sentence:
		if tag != None and tag.startswith('V'):
			print '\t\tVerb: ', word
			verbs.append(word)
			if verbFound == False: verbFound = True
	if not verbFound:
		print '\t\t *** No Verbs Found ***'
		verbs = None


# ****** Finding most Interesting Collocations ******

# from nltk.collocations import *

# trainingdata_filename = 'traintweets_unlabeled.csv'

# bigram_measures = nltk.collocations.BigramAssocMeasures()
# trigram_measures = nltk.collocations.TrigramAssocMeasures()

# # change this to read in your data
# finder = BigramCollocationFinder.from_words(
#    trainingdata_filename)

# # only bigrams that appear 3+ times
# finder.apply_freq_filter(3) 

# # return the 10 n-grams with the highest PMI
# highest_pmi = finder.nbest(bigram_measures.pmi, 10)  
# print highest_pmi

# tag_fd = nltk.FreqDist((word, tag) in candidate_tagged_words)
# print 'keys are... ', tag_fd.keys()
# print 'values are... ', tag_fd.values()

# word_tag_fd = nltk.FreqDist(candidate_tagged_words)
# print [word + "/" + tag for (word, tag) in word_tag_fd if tag.startswith('V')]