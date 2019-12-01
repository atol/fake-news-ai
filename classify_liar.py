import random
import pickle
import pandas as pd
import numpy as np

# Load pre-trained claim model
with open('models/liar_claims_tfidf_svm.pkl', 'rb') as f:
    model= pickle.load(f)

# Multinomial Naive Bayes classifier that classifies a claim as 
# 0, 1 or 2 based on a learned model trained on the claim text
def classify_claims(cl):
    # Get the claim from the json entry
    claim = cl['claim']
    # Convert to a list to avoid iterable error
    claim = [claim]
    # Predict the label using the learned model
    pred = model.predict(claim)
    return pred[0] # pred returns a numpy array of size 1, so get value at index 0