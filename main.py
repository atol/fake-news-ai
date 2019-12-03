import json
import os
from fnc import *
from sklearn.metrics import f1_score
from keras.models import load_model
from keras.optimizers import SGD

# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = 'dataset/metadata.json'
ARTICLES_FILEPATH = 'dataset/articles'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = 'output/predictions.txt'

if __name__ == '__main__':
    # Load pre-trained model
    model = load_model('models/liar_claims_conv1d_e100.h5')
    model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, nesterov=True),
              metrics=['acc'])

    # Load tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Read in the metadata file.
    with open(METADATA_FILEPATH, 'r') as f:
        claims = json.load(f)
    
    # Create a predictions file.
    print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
    with open(PREDICTIONS_FILEPATH, 'w') as f:
        for claim in claims:
            f.write('%d,%d\n' % (claim['id'], make_prediction(claim, model, tokenizer)))
    print('Finished writing predictions.')