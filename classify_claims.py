import os
import json
import random
import pickle
import pandas as pd
import numpy as np

# Load false claim model
with open('models/train_false_claims.pkl', 'rb') as f:
    model_false = pickle.load(f)

# Load partly true claim model
with open('models/train_partly_claims.pkl', 'rb') as f:
    model_partly = pickle.load(f)

# Load true claim model
with open('models/train_true_claims.pkl', 'rb') as f:
    model_true = pickle.load(f)

# Get the list of classifiers
def get_classifiers():
    classifiers = [ classify_false, classify_partly, classify_true ]
    return classifiers

# Predicts whether a claim is false or not based on a pre-trained model
# Returns 0 if the claim is false and -1 otherwise
def classify_false(cl):
    # Get the claim from the json entry
    claim = cl['claim']
    # Convert to a list to avoid iterable error
    claim = [claim]
    # Predict the label using the learned model
    pred = model_false.predict(claim)
    print("False classifier prediction:", pred)
    return pred[0] # pred returns a numpy array of size 1, so get value at index 0

# Predicts whether a claim is partly true or not based on a pre-trained model
# Returns 1 if the claim is partly true and -1 otherwise
def classify_partly(cl):
    # Get the claim from the json entry
    claim = cl['claim']
    # Convert to a list to avoid iterable error
    claim = [claim]
    # Predict the label using the learned model
    pred = model_partly.predict(claim)
    print("Partly classifier prediction:", pred)
    return pred[0] # pred returns a numpy array of size 1, so get value at index 0

# Predicts whether a claim is true or not based on a pre-trained model
# Returns 2 if the claim is true and -1 otherwise
def classify_true(cl):
    # Get the claim from the json entry
    claim = cl['claim']
    # Convert to a list to avoid iterable error
    claim = [claim]
    # Predict the label using the learned model
    pred = model_true.predict(claim)
    print("True classifier prediction:", pred)
    return pred[0] # pred returns a numpy array of size 1, so get value at index 0

def voting_classifier(cl):
    # Initialize label counts
    false_count = 0
    partly_count = 0
    true_count = 0

    # Get classifiers
    classifiers = get_classifiers()

    # Loop through classifiers and evaluate the passed-in claim
    for i in range(0, len(classifiers)):
        clf = classifiers[i]

        # Get prediction from classifier
        pred = clf(cl)

        # Multiply prediction by weight and add to appropriate tally
        if pred == 0:
            false_count += 1
        elif pred == 1:
            partly_count += 1
        else:
            true_count += 1
    
    print("False:", false_count)
    print("Partly true:", partly_count)
    print("True:", true_count)

    best = max([false_count, partly_count, true_count])

    if best == false_count:
        return 0
    elif best == partly_count:
        return 1
    else:
        return 2