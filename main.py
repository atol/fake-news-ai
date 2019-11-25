import json
import os
import pickle
import pandas as pd

# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = 'dataset/metadata.json'
ARTICLES_FILEPATH = 'dataset/articles'

# This is the filepath where the predictions should be written to.
PREDICTIONS_FILEPATH = 'predictions.txt'

if __name__ == '__main__':
    # Load learned model
    with open('models/combined_claim_claimant.pkl', 'rb') as f:
        model = pickle.load(f)

    # Read in the metadata file
    with open(METADATA_FILEPATH, 'r') as f:
        data = json.load(f)
    
    # Convert metadata file to dataframe
    df = pd.DataFrame(data)

    # Combine claim and claimant
    df["combined"] = df["claim"].map(str) + ' ' + df["claimant"]
    X = df.combined
    y = df.label

    # Use model to make predictions
    pred = model.predict(X)

    # Join predictions to dataframe
    df['prediction'] = pred

    # Create a predictions file
    print('\nWriting predictions to:', PREDICTIONS_FILEPATH)
    with open(PREDICTIONS_FILEPATH, 'w') as f:
        for index, row in df.iterrows():
            f.write('%d,%d\n' % (row['id'], row['prediction']))
    print('Finished writing predictions.')