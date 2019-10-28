import os
import json
import random
import pickle
import pandas as pd
import numpy as np

from statistics import mode
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

FALSE = 0
PARTLY = 1
TRUE = 2

# Looked at the claims made by each claimant in the training data and
# tallied the number of true, partly, and false claims made by each claimant.
# Each claimant was assigned a label of 0, 1, 2 depending on whether the majority 
# of their claims were false, true or partly. In the event of a tie, the lower
# value label (0 < 1 < 2) was chosen.
# TODO: Use a probability distribution instead?
# TODO: Give some rating to anonymous claims
fnc_claimants = pickle.load( open( "input/train_claimants.p", "rb" ) )

# Looked at the related_article field of each claim in the training data. 
# For each article in the claim's related articles, checked to see which other
# claims had cited that article. Then looked at whether the associated claims 
# were labelled false, partly or true. Each article_id was assigned a label of
# 0, 1 or 2 depending on whether the majority of the claims citing the article
# was labelled false, partly or true. In the event of a tie, the lower
# value label (0 < 1 < 2) was chosen.
# TODO: Use a probability distribution instead?
fnc_article_ids = pickle.load( open( "input/train_article_ids.p", "rb" ) )

# Get the list of classifiers
def get_classifiers():
    classifiers = [ classify_weighted_random, classify_claim_len, classify_related_count, 
                    classify_word_count, classify_claimant, classify_related_article_id ]
    return classifiers

# Get the weights for the classifiers
def get_weights(classifiers, metric):
    # Load correctly labelled claims from training data
    with open('input/train.json', 'r') as f:
        dev = json.load(f)
    
    # Calculate weights for each classifier according to the given metric
    weights = []
    for clf in classifiers:
        w = metric(dev, clf)
        # If weight is negative, set to 0
        if w < 0:
            w = 0
        weights.append(w)
    
    # F1 score: {'weighted_random': 0.3320465510889827, 'claim_len': 0.24490294435522805, 'related_count': 0.27568939517104935
    #            'word_count': 0.06978862049083771, 'claimant': 0.5949752273857106, 'related_article_id': 0.9587703259365107}
    # Matthews correlation coefficient score: 
    #           {'weighted_random': 0.005098654305867974, 'claim_len': 0.015171210964371292, 'related_count': -0.01516493989932876
    #            'word_count': 0.003950039080365395, 'claimant': 0.49405385976289373, 'related_article_id': 0.9449845286946165}

    return weights

# Takes a classifier and returns its F1 score
# F1 score = (2 * precision * recall) / (precision + recall)
def eval_f1(claims, classifier):
    true = []
    pred = []
    # Loop through claims
    for cl in claims:
        # Append correct claim label to 'true'
        true.append(cl['label'])
        # Append label predicted by classifier to 'pred'
        pred.append(classifier(cl))
    # f1_score compares the predicted values to the true values
    # average='macro' calculate metrics for each label and find their unweighted mean
    # labels=np.unique(pred) only shows labels that were predicted at least once (suppresses warning)
    result = f1_score(true, pred, average='macro', labels=np.unique(pred))
    return result

# Takes a classifier and returns its Matthews correlation coefficient (MCC) score
# MCC score = 
def eval_mcc(claims, classifier):
    true = []
    pred = []
    # Loop through claims
    for cl in claims:
        # Append correct claim label to 'true'
        true.append(cl['label'])
        # Append label predicted by classifier to 'pred'
        pred.append(classifier(cl))
    # matthews_corrcoef compares the predicted values to the true values
    result = matthews_corrcoef(true, pred)
    return result

# Computes a confusion matrix for a given classifier and set of claims
def conf_matrix(claims, classifier):
    conf = [[0,0,0], [0,0,0], [0,0,0]]
    for cl in claims:
        prediction = classifier(cl)
        conf[cl['label']][prediction] += 1
    return conf

# Takes a classifier and returns a tuple of (right, wrong) predictions
def eval_conf(claims, classifier):
    conf = conf_matrix(claims, classifier)
    right = conf[TRUE][TRUE] + conf[PARTLY][PARTLY] + conf[FALSE][FALSE]
    wrong = sum(sum(lst) for lst in conf) - right
    return (right/len(claims)) # TODO: Need to find an evaluation metric

# Prints out a confusion matrix
def print_conf(title, classifier):
    conf = conf_matrix(claims, classifier)
    print()
    print(title + ':')
    def fmt(n): return str(n).ljust(5)
    print('        actual')
    print('   ' + fmt('t'), fmt('p'), fmt('f'))
    print('t ', fmt(conf[TRUE][0]), fmt(conf[TRUE][1]), fmt(conf[TRUE][2]))
    print('p ', fmt(conf[PARTLY][0]), fmt(conf[PARTLY][1]), fmt(conf[PARTLY][2]))
    print('f ', fmt(conf[FALSE][0]), fmt(conf[FALSE][1]), fmt(conf[FALSE][2]))

    right = conf[TRUE][TRUE] + conf[PARTLY][PARTLY] + conf[FALSE][FALSE]
    wrong = sum(sum(lst) for lst in conf) - right
    print()
    print(f'  right: {right} ({100*right/len(claims):.2f}%)')
    print(f'  wrong: {wrong} ({100*wrong/len(claims):.2f}%)')

def classify_uniform_random(cl): return random.randint(0, 2)
def classify_all_true(cl): return 2
def classify_all_partly(cl): return 1
def classify_all_false(cl): return 0

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
    # assign 0 (false) because it's the most frequently occurring label
    # TODO: Return probability distribution?
    else:
        return 0
    
# Returns 0, 1, 2 depending on the article IDs of the claim's related articles
def classify_related_article_id(cl):
    related = []
    # Loop through the claim's related articles
    for article in cl['related_articles']:
        # If the article ID was seen in the training data,
        # return the label associated with the article
        if article in fnc_article_ids:
            related.append(fnc_article_ids[article])
        # Otherwise, if the article was not seen in the training data,
        # assign 0 (false) because it's the most frequently occuring label
        # TODO: Return probability distribution?
        else:
            return 0
    # Return 0, 1, 2 depending on whether the majority of the claim's
    # related articles are labeled false, partly or true
    # TODO: Use probability distribution?
    try:
        pred = mode(related)
    # In the event of a tie, return the lowest value label
    except:
        pred = min(related)
    return pred

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

def weighted_voting_classifier(claim, classifiers, weights):
    # Initialize label counts
    false_count = 0
    partly_count = 0
    true_count = 0

    # Loop through classifiers and evaluate the passed-in claim
    for i in range(0, len(classifiers)):
        clf = classifiers[i]
        w = weights[i]

        # Get prediction from classifier
        pred = clf(claim)

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