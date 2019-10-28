import json
from classifiers import *
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import numpy as np

# F1 score for predictions.txt
def f1(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('output/train.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = f1_score(true, pred, average='macro')
    return result

# Matthews correlation coefficient score for predictions.txt
def mcc(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('output/train.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = matthews_corrcoef(true, pred)
    return result

if __name__ == '__main__':
    # Set up classifiers and weights
    classifiers = get_classifiers()
    weights = get_weights(classifiers, eval_f1)

    # Read in the metadata file
    with open("input/train.json", 'r') as f:
        claims = json.load(f)
    
    # Create a predictions file
    print('\nWriting predictions to:', "output/train.txt")
    with open("output/train.txt", 'w') as f:
        for claim in claims:
            f.write('%d,%d\n' % (claim['id'], weighted_voting_classifier(claim, classifiers, weights)))
    print('Finished writing predictions.')
    
    # Compute results
    print("\nF1 score:", f1(claims))
    print("\nMatthews correlation coefficient:", mcc(claims))