import pickle

# Load learned model
with open('/usr/src/models/combined_claim_claimant_oversampled.pkl', 'rb') as f:
    model = pickle.load(f)

def classify_combined(cl):
    # Get the claim and claimant from the json entry
    claim = cl['claim']
    claimant = cl['claimant']
    sequence = claim + ' ' + claimant
    # Convert to a list to avoid iterable error
    sequence = [sequence]
    # Predict the label using the learned model
    pred = model.predict(sequence)
    return pred[0] # pred returns a numpy array of size 1, so get value at index 0