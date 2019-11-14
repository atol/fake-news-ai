import os
import json
import random
import pickle
import pandas as pd
import numpy as np

from statistics import mode
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

import textblob
from textblob import TextBlob

FALSE = 0
PARTLY = 1
TRUE = 2

# # Looked at the claims made by each claimant in the training data and
# # tallied the number of true, partly, and false claims made by each claimant.
# # Each claimant was assigned a label of 0, 1, 2 depending on whether their
# # claims were, on average, most likely to be false, true or partly.
# fnc_claimants = pickle.load( open( "input/train_claimants.p", "rb" ) )

# # Looked at the related_article field of each claim in the training data. 
# # For each article in the claim's related articles, checked to see which other
# # claims had cited that article. Then looked at whether the associated claims 
# # were labelled false, partly or true. Each article_id was assigned a label of
# # 0, 1 or 2 depending on whether the claims citing the article were, on average,
# # most likely to be false, partly or true.
# fnc_article_ids = pickle.load( open( "input/train_articles1.p", "rb" ) )

# Load pre-trained claim model
with open('models/train_claims_nb.pkl', 'rb') as f:
    clf_claims = pickle.load(f)

# Load pre-trained claimant model
with open('models/train_claimants_nb.pkl', 'rb') as f:
    clf_claimants = pickle.load(f)

# Load pre-trained related article model
with open('models/train_articles_nb.pkl', 'rb') as f:
    clf_articles = pickle.load(f)

# Get the list of classifiers
def get_classifiers():
    classifiers = [ classify_claims, classify_claimants, classify_related_articles ]
    return classifiers

# Get the weights for the classifiers
def get_weights(classifiers, metric, articles):
    # Load correctly labelled claims from training data
    with open('input/train.json', 'r') as f:
        dev = json.load(f)
    
    # Calculate weights for each classifier according to the given metric
    weights = []
    for clf in classifiers:
        w = metric(dev, clf, articles)
        # If weight is negative, set to 0
        if w < 0:
            w = 0
        weights.append(w)

    return weights

# Takes a classifier and returns its F1 score
# F1 score = (2 * precision * recall) / (precision + recall)
def eval_f1(claims, classifier, articles):
    true = []
    pred = []
    # Loop through claims
    for cl in claims:
        # Append correct claim label to 'true'
        true.append(cl['label'])
        # Append label predicted by classifier to 'pred'
        pred.append(classifier(cl, articles))
    # f1_score compares the predicted values to the true values
    # average='macro' calculate metrics for each label and find their unweighted mean
    result = f1_score(true, pred, average='macro')
    return result

# Takes a classifier and returns its Matthews correlation coefficient (MCC) score
# MCC score = 
def eval_mcc(claims, classifier, articles):
    true = []
    pred = []
    # Loop through claims
    for cl in claims:
        # Append correct claim label to 'true'
        true.append(cl['label'])
        # Append label predicted by classifier to 'pred'
        pred.append(classifier(cl, articles))
    # matthews_corrcoef compares the predicted values to the true values
    result = matthews_corrcoef(true, pred)
    return result

def eval_acc(claims, classifier, articles):
    true = []
    pred = []
    # Loop through claims
    for cl in claims:
        # Append correct claim label to 'true'
        true.append(cl['label'])
        # Append label predicted by classifier to 'pred'
        pred.append(classifier(cl, articles))
    # matthews_corrcoef compares the predicted values to the true values
    result = accuracy_score(true, pred)
    return result

# Randomly chooses a number from (0, 1, 2) corresponding to the (false, partly, true) label.
# Each label has a weight determined by its percentage in the training data.
# False claims: 47.62%
# Partly claims: 41.47%
# True claims: 10.90%
def classify_weighted_random(cl):
    pred = [0] * 48 + [1] * 41 + [2] * 11
    return random.choice(pred)

# false:  {'min': 15, 'max': 7251, 'mean': 142.4341252699784, 'median': 101.0, 'pstdev': 252.27116040305341}
# partly: {'min': 22, 'max': 8441, 'mean': 140.84715547977058, 'median': 111, 'pstdev': 196.08532387439064}
# true:   {'min': 20, 'max': 5716, 'mean': 124.96403301886792, 'median': 104.0, 'pstdev': 172.86696816567172}
def classify_claim_len(cl):
    size = len(cl['claim'])
    if size >= (142.34 + 140.85) / 2:
        return 2
    elif size >= (1440.85 + 124.96) / 2:
        return 1
    else:
        return 0

# false:  {'min': 2, 'max': 66, 'mean': 5.262688984881209, 'median': 4.0, 'pstdev': 4.43623540426079}
# partly: {'min': 2, 'max': 41, 'mean': 4.841574949620214, 'median': 4, 'pstdev': 3.2703837621469978}
# true:   {'min': 2, 'max': 27, 'mean': 4.399174528301887, 'median': 3.0, 'pstdev': 3.0169293207073387}
def classify_related_count(cl):
    size = len(cl['related_articles'])
    if size >= (5.26 + 4.84) / 2:
        return 2
    elif size >= (4.84 + 4.40) / 2:
        return 1
    else:
        return 0

# Total # of claims:
#    0=7408, 1=6451, 2=1696
#
# Avg # words per claim:
#    0=23.4, 1=23.1, 2=20.6
#
# Avg word length per claim:
#    0=5.2, 1=5.2, 2=5.2
#
def classify_word_count(cl):
    size = len(cl['claim'].split())
    if size >= (5.26 + 4.84) / 2:
        return 2
    elif size >= (4.84 + 4.40) / 2:
        return 1
    else:
        return 0

# Returns 0, 1 or 2 depending on the claimant
def classify_claimant(cl):
    # If the claimant was seen in the training data,
    # return the label associated with the claimant
    if cl['claimant'] in fnc_claimants:
        return fnc_claimants[cl['claimant']]
    # Otherwise, if the claimant was not seen in the training data,
    # return weighted random value
    else:
        return classify_weighted_random(cl)
    
# Returns 0, 1, 2 depending on the article IDs of the claim's related articles
def classify_related_article_id(cl):
    related = []
    # Loop through the claim's related articles
    for article in cl['related_articles']:
        # If the article ID was seen in the training data,
        # return the label associated with the article
        if article in fnc_article_ids:
            labels = fnc_article_ids[article]
            score = round(sum(labels)/len(labels))
            related.append(score)
        # Otherwise, if the article was not seen in the training data,
        # return weighted random value
        else:
            return classify_weighted_random(cl)
    # Return average label of the claim's related articles
    return round(sum(related)/len(related))

# Returns 0, 1 or 2 depending on the claim's subjectivity score
# Subjectivity score is a float within the range [0.0, 1.0]
# where 0.0 is very objective and 1.0 is very subjective.
def classify_subjectivity(cl):
    claim = TextBlob(cl['claim'])
    subjectivity = claim.sentiment.subjectivity
    if subjectivity > 0.75:
        return 0
    elif subjectivity < 0.2:
        return 2
    else:
        return 1

# Returns 0, 1 or 2 depending on the claim's polarity score
# Polarity score is a float within the range [-1.0, 1.0]
# where -1.0 is very negative and 1.0 is very positive
def classify_polarity(cl):
    claim = TextBlob(cl['claim'])
    polarity = claim.sentiment.polarity
    if polarity <= -0.75 or polarity >= 0.75:
        return 0
    elif polarity >= -0.1 and polarity <= 0.1:
        return 2
    else:
        return 1

# Multinomial Naive Bayes classifier that classifies a claim as 
# 0, 1 or 2 based on a learned model trained on the claim text
def classify_claims(cl, articles):
    # Get the claim from the json entry
    claim = cl['claim']
    # Convert to a list to avoid iterable error
    claim = [claim]
    # Predict the label using the learned model
    pred = clf_claims.predict(claim)
    return pred[0] # pred returns a numpy array of size 1, so get value at index 0

# Multinomial Naive Bayes classifier that classifies a claim as 
# 0, 1 or 2 based on a learned model trained on the claim text
def classify_claimants(cl, articles):
    # Get the claimant from the json entry
    claimant = cl['claimant']
    # Convert to a list to avoid iterable error
    claimant = [claimant]
    # Predict the label using the learned model
    pred = clf_claimants.predict(claimant)
    return pred[0] # pred returns a numpy array of size 1, so get value at index 0

# Returns 0, 1, 2 depending on the content of the claim's related articles
def classify_related_articles(cl, articles):
    labels = []
    # Loop through the claim's related articles
    for article_id in cl['related_articles']:
        # Get article from dictionary
        article = articles[str(article_id)]
        # Predict label
        pred = clf_articles.predict([article])
        labels.append(pred[0])
    # Return average label of the claim's related articles
    return round(sum(labels)/len(labels))

def voting_classifier(claim, classifiers):
    # Each classifier returns a 'vote' for the claim's label
    votes = [clf(claim) for clf in classifiers]
    # Pick the most frequently voted label
    try:
        pred = mode(votes)
    # Or in the event of a tie, pick the smallest label value
    except:
        pred = min(votes)
    return pred

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def weighted_voting_classifier(claim, classifiers, weights, articles):
    # Initialize label counts
    false_count = 0
    partly_count = 0
    true_count = 0

    # Loop through classifiers and evaluate the passed-in claim
    for i in range(0, len(classifiers)):
        clf = classifiers[i]
        w = weights[i]

        # Get prediction from classifier
        pred = clf(claim, articles)

        # Multiply prediction by weight and add to appropriate tally
        if pred == 0:
            false_count += 1*w
        elif pred == 1:
            partly_count += 1*w
        else:
            true_count += 1*w
    
    best = max([false_count, partly_count, true_count])

    if best == false_count:
        return 0
    elif best == partly_count:
        return 1
    else:
        return 2