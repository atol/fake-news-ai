import json
from classifiers import *
from sklearn.metrics import f1_score

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

if __name__ == '__main__':
    with open(METADATA_FILEPATH, 'r') as f:
        claims = json.load(f)

    print("dev score:", f1(claims))