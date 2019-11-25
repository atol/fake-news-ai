import json
import glob
from classify_claims import *
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score

ARTICLES_FILEPATH = 'dataset/articles'

# F1 score for predictions.txt
def f1(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('output/test.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = f1_score(true, pred, average='macro')
    return result

# Matthews correlation coefficient score for predictions.txt
def mcc(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('output/test.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = matthews_corrcoef(true, pred)
    return result

if __name__ == '__main__':
    # Read in the metadata file
    with open("input/test.json", 'r') as f:
        claims = json.load(f)
    
    claim = claims[2]

    # Create a predictions file
    print('Writing predictions to:', "output/test.txt")
    with open("output/test.txt", 'w') as f:
        f.write('%d,%d\n' % (claim['id'], voting_classifier(claim)))
    print('Finished writing predictions.')
    
    # # Compute results
    # print("\nF1 score:", f1(claims))