import json
from classifiers import *
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import numpy as np

METADATA_FILEPATH = 'dataset/metadata.json'
ARTICLES_FILEPATH = 'dataset/articles'

# F1 score for predictions.txt
def f1(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('answer/predictions.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = f1_score(true, pred, average='macro')
    return result

# F1 score for individual classifiers
def f1_clf(claims, classifier):
    true = []
    pred = []
    for cl in claims:
        true.append(cl['label'])
        pred.append(classifier(cl))
    result = f1_score(true, pred, average='macro')
    return result

# Matthews correlation coefficient score for predictions.txt
def mcc(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('answer/predictions.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = matthews_corrcoef(true, pred)
    return result

# Matthews correlation coeeficient score for individual classifiers
def mcc_clf(claims, classifier):
    true = []
    pred = []
    for cl in claims:
        true.append(cl['label'])
        pred.append(classifier(cl))
    result = matthews_corrcoef(true, pred)
    return result

if __name__ == '__main__':
    with open(METADATA_FILEPATH, 'r') as f:
        claims = json.load(f)

    classifiers = [classify_weighted_random, classify_claim_len, classify_related_count, classify_word_count, classify_claimant, classify_related_article_id]
    
    print("f1 score:", f1(claims))
    print("weighted random:", f1_clf(claims, classify_weighted_random))
    print("claim length:", f1_clf(claims, classify_claim_len))
    print("related count:", f1_clf(claims, classify_related_count))
    print("word count:", f1_clf(claims, classify_word_count))
    print("claimaint:", f1_clf(claims, classify_claimant))
    print("related articles:", f1_clf(claims, classify_related_article_id))

    print("matthews score:", mcc(claims))
    print("weighted random:", mcc_clf(claims, classify_weighted_random))
    print("claim length:", mcc_clf(claims, classify_claim_len))
    print("related count:", mcc_clf(claims, classify_related_count))
    print("word count:", mcc_clf(claims, classify_word_count))
    print("claimaint:", mcc_clf(claims, classify_claimant))
    print("related articles:", mcc_clf(claims, classify_related_article_id))