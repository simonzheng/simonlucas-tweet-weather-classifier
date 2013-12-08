from cspclassifier import CSPClassifier


classifier = CSPClassifier('data/train.csv')
classifier.evaluate(5)