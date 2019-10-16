import json
import os
from classifiers import *

# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = 'dataset/metadata.json'
ARTICLES_FILEPATH = 'dataset/articles'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = 'answer/predictions.txt'

if __name__ == '__main__':
# Read in the metadata file.
    with open(METADATA_FILEPATH, 'r') as f:
        claims = json.load(f)

    classifiers = [classify_weighted_random, classify_claim_len, classify_word_count, classify_claimant, classify_related_article_id]

    # Create a predictions file.
    print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
    with open(PREDICTIONS_FILEPATH, 'w') as f:
        for claim in claims:
            prediction = weighted_voting_classifier(claim, classifiers)
            f.write('%d,%d\n' % (claim['id'], prediction) )
    print('Finished writing predictions.')