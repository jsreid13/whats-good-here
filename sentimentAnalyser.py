from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize

import os
import random
import pickle
from statistics import mode


class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers[0]:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers[0]:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# Find the which words in the review are contained within the word_features
# what were determined from the movie review dataset
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features


# Load a pickle file, if not found return false
def load_pickle(pickleName: str):
    with open(os.getcwd() + "/%s"
              % pickleName, "rb") as trainedDataset:
        #  print("Loaded pickle file %s" % pickleName)
        return pickle.load(trainedDataset)


# Sentiment analysis using NLTK movie reviews corpus to train
documents = load_pickle("documents.pickle")
random.shuffle(documents)
word_features = load_pickle("wordFeats5k.pickle")

# featureSets = load_pickle("featureSet.pickle")
# trainingSet = featureSets[:10000]
# testingSet = featureSets[10000:]

# If the pickle files containing trained data exists then load it and
# move on, otherwise train the sentiment analysis and save it
classifier = load_pickle("naivebayes.pickle")
# print("Naive Bayes Algo Accuracy percent: ",
#       (nltk.classify.accuracy(classifier, testingSet))*100)
# classifier.show_most_informative_features(15)


scikitClassifiers = ["MultinomialNB.pickle"
                     , "BernoulliNB.pickle"
                     , "LogisticRegression.pickle"
                     , "SGDClassifier.pickle"
                     , "LinearSVC.pickle"
                     , "NuSVC.pickle"
                     ]

allClassifiers = [classifier]
for c in scikitClassifiers:
    allClassifiers.append(load_pickle(c))

voted_classifier = VoteClassifier(allClassifiers)
#  print(str(voted_classifier) + " is best")
# print("voted_classifier accuracy percent: ",
#       (nltk.classify.accuracy(voted_classifier, trainingSet))*100)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
