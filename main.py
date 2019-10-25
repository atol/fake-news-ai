import json
import os
from classifiers import *

# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = 'dataset/metadata.json'
ARTICLES_FILEPATH = 'dataset/articles'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = 'predictions.txt'

if __name__ == '__main__':
    # Set up classifiers and weights
    classifiers = get_classifiers()
    weights = get_weights(classifiers, eval_mcc)

    # Read in the metadata file.
    with open(METADATA_FILEPATH, 'r') as f:
        claims = json.load(f)
    
    # Create a predictions file.
    print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
    with open(PREDICTIONS_FILEPATH, 'w') as f:
        for claim in claims:
            f.write('%d,%d\n' % (claim['id'], weighted_voting_classifier(claim, classifiers, weights)))
    print('Finished writing predictions.')