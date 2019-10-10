import json
import os
import random

# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = '/usr/local/dataset/metadata.json'
ARTICLES_FILEPATH = '/usr/local/dataset/articles'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = '/usr/local/predictions.txt'

# Weighted random selection based on probability of label in training set
def weighted_random():
    label = [0] * 50 + [1] * 40 + [2] * 10
    return random.choice(label)

if __name__ == '__main__':
# Read in the metadata file.
    with open(METADATA_FILEPATH, 'r') as f:
        claims = json.load(f)

    # Create a predictions file.
    print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
    with open(PREDICTIONS_FILEPATH, 'w') as f:
        for claim in claims:
            f.write('%d,%d\n' % (claim['id'], weighted_random()) )
    print('Finished writing predictions.')