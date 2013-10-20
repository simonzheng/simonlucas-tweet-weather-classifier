import sys
import getopt
import os
import math

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10

    # These are my defined variables
    self.posWordCounts = {}
    self.negWordCounts = {}
    self.posWords = set()
    self.negWords = set()
    self.vocab = set()
    self.numPosDocs = 0.0
    self.numNegDocs = 0.0
    self.numPosTokens = 0.0
    self.numNegTokens = 0.0


  #############################################################################
  # TODO TODO TODO TODO TODO 
  
  def classify(self, words):
    # print "count(good|pos) = %f" %self.posWordCounts['good']
    # print "count(good|neg) = %f" %self.negWordCounts['good']
    # print "total pos = %f" %self.numPosTokens
    # print "total neg = %f" %self.numNegTokens
    # print 'probability of good given positive is %f' % (self.posWordCounts['good'] / self.numPosTokens)
    # print 'probability of good given negative is %f' % (self.negWordCounts['good'] / self.numNegTokens)
    
    
    # print 'the most frequent words in positive docs are'
    # posWordFreqs = self.posWordCounts.values()
    # posWordFreqs.sort()
    # for i in range(10):
    #   minPosWordFreq = posWordFreqs[len(posWordFreqs) - 1 - i]
    #   print minPosWordFreq
    # for (key,value) in self.posWordCounts.items():
    #   if value >= minPosWordFreq:
    #     print (key,value)

    # print 'the most frequent words in negative docs are'
    # negWordFreqs = self.negWordCounts.values()
    # negWordFreqs.sort()
    # for i in range(10):
    #   minNegWordFreq = negWordFreqs[len(negWordFreqs) - 1 - i]
    #   print minNegWordFreq
    # for (key,value) in self.negWordCounts.items():
    #   if value >= minNegWordFreq:
    #     print (key,value)


    totalNumDocs = self.numPosDocs + self.numNegDocs
    posScore = math.log(self.numPosDocs / totalNumDocs)
    negScore = math.log(self.numNegDocs / totalNumDocs)

    vocabSize = len(self.vocab)

    for word in words:
      if word in self.posWordCounts:
        posScore += math.log( (self.posWordCounts[word]+1.0) / (self.numPosTokens + vocabSize))
      else:
        posScore += math.log( 1.0 / (self.numPosTokens + vocabSize))

      if word in self.negWordCounts:
        negScore += math.log( (self.negWordCounts[word]+1.0) / (self.numNegTokens + vocabSize))
      else:
        negScore += math.log( 1.0 / (self.numNegTokens + vocabSize))

    if (posScore > negScore): 
      # print 'returning pos and posScore is %f and negScore is %f' % (posScore, negScore)
      return 'pos'
    else:
      # print 'returning neg and posScore is %f and negScore is %f' % (posScore, negScore)
      return 'neg'
  
  def addExample(self, klass, words):
    for word in words:
      # add the word to the vocabulary list
      if (not (word in self.vocab)): 
        self.vocab.add(word)

      # add the word to the number of positive or negative tokens and add to counts of words
      if (klass == 'pos'):
        self.numPosTokens += 1.0
        if word in self.posWords:
          self.posWordCounts[word] += 1.0
        else:
          self.posWords.add(word)
          self.posWordCounts[word] = 1.0
      else:
        self.numNegTokens += 1.0
        if word in self.negWords:
          self.negWordCounts[word] += 1.0
        else:
          self.negWords.add(word)
          self.negWordCounts[word] = 1.0

    # increase the number of docs
    if (klass == 'pos'): 
      self.numPosDocs += 1.0
    else:
      self.numNegDocs += 1.0
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)

  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits


  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      guess = self.classify(words)
      labels.append(guess)
    return labels
  
  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = [] 
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1: 
      print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName)) 
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName)) 
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    accuracy = 0.0
    for example in split.train:
      words = example.words
      if FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      classifier.addExample(example.klass, words)
  
    for example in split.test:
      words = example.words
      if FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyFile(FILTER_STOP_WORDS, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print classifier.classify(testFile)
    
def main():
  FILTER_STOP_WORDS = False
  (options, args) = getopt.getopt(sys.argv[1:], 'f')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  
  if len(args) == 2 and os.path.isfile(args[1]):
    classifyFile(FILTER_STOP_WORDS, args[0], args[1])
  else:
    test10Fold(args, FILTER_STOP_WORDS)

if __name__ == "__main__":
    main()
