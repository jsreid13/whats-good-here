import nltk
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

import time
import random
import pickle
import os
from statistics import mode


class VoteClassifier(ClassifierI):

    def __init__(self, classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# Find the which words in the review are contained within the word_features
# what were determined from the movie review dataset
def find_features(_document, _word_features):
    words = word_tokenize(_document)
    features = {}
    for word in _word_features:
        features[word] = (word in words)
    return features


# Use nltk to tag words with what type of word they are and
# return all nouns
def isAdjOrAdv(pos): return pos[:2].startswith(("J", "R"))


# Sentiment analysis using NLTK movie reviews corpus to train
# Create feature set containing the words used most in movie_reviews
short_pos = open(os.getcwd() + "/short_reviews/positive.txt").read()
short_neg = open(os.getcwd() + "/short_reviews/negative.txt").read()
documents = []
all_words = []
for entry in short_pos.split('\n'):
    documents.append( (entry, "pos") )
    words = word_tokenize(entry)
    POS = nltk.pos_tag(words)
    [all_words.append(w[0].lower()) for w in POS if isAdjOrAdv(w[1])]
for entry in short_neg.split('\n'):
    documents.append( (entry, "neg") )
    words = word_tokenize(entry)
    POS = nltk.pos_tag(words)
    [all_words.append(w[0].lower()) for w in POS if isAdjOrAdv(w[1])]
#    for category in movie_reviews.categories():
#        for fileid in movie_reviews.fileids(category):
#            documents.append((list(movie_reviews.words(fileid)), category))
#    [all_words.append(w.lower()) for w in movie_reviews.words()]
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]
featureSets = [(find_features(rev, word_features), category) for (rev, category) in documents]
random.shuffle(featureSets)

# Write processed training data
with open(os.getcwd() + "documents.pickle",
          "wb") as trainingDataset:
    pickle.dump(documents, trainingDataset)
with open(os.getcwd() + "/wordFeats5k.pickle",
          "wb") as trainingDataset:
    pickle.dump(word_features, trainingDataset)
with open(os.getcwd() + "/featureSet.pickle",
          "wb") as trainingDataset:
    pickle.dump(featureSets, trainingDataset)

print("STOP")
trainingSet = featureSets[:10000]
testingSet = featureSets[10000:]
# If the pickle files containing trained data exists then load it and
# move on, otherwise train the sentiment analysis and save it
classifier = nltk.NaiveBayesClassifier.train(trainingSet)
print("NaiveBayesClassifier accuracy %",
      (nltk.classify.accuracy(classifier, testingSet))*100)
with open(os.getcwd() + "/naivebayes.pickle",
          "wb") as save_classifier:
    pickle.dump(classifier, save_classifier)

print("Naive Bayes Algo Accuracy percent: ",
      (nltk.classify.accuracy(classifier, testingSet))*100)
classifier.show_most_informative_features(15)


# Trains the different classifiers in SklearnClassifier's arsenal
def trainer(clr, trainingSet):
    classifier = SklearnClassifier(clr)
    classifier.train(trainingSet)
    print(str(classifier) + "Algo Accuracy percent: ",
          (nltk.classify.accuracy(classifier, testingSet))*100)
    return classifier

scikitClassifiers = [MultinomialNB()
                     , BernoulliNB()
                     , LogisticRegression()
                     , SGDClassifier()
                     , LinearSVC()
                     , NuSVC()
                     ]

allClassifiers = [classifier]
for c in scikitClassifiers:
    trained_clf = trainer(c, trainingSet)
    allClassifiers.append(trained_clf)
    with open(os.getcwd() + "/%s.pickle"
              % str(c).split('(')[0], "wb") as save_classifier:
        pickle.dump(trained_clf, save_classifier)

voted_classifier = VoteClassifier(allClassifiers)
print("voted_classifier accuracy percent: ",
      (nltk.classify.accuracy(voted_classifier, trainingSet))*100)
print("Classification: ", voted_classifier.classify(testingSet[0][0]),
      "Confidence: ", voted_classifier.confidence(testingSet[0][0]))
print(testingSet[0][0])
print("Classification: ", voted_classifier.classify(testingSet[1][0]),
      "Confidence: ", voted_classifier.confidence(testingSet[1][0]))
print(testingSet[1][0])

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats)
