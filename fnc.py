import pickle
import string
import gensim
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 1000

def clean_text(line):
    text = []
    tokens = word_tokenize(line)
    # Convert to lowercase
    tokens = [w.lower() for w in tokens]
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # Remove numbers
    words = [word for word in stripped if word.isalpha()]
    text.append(words)
    return text

def preprocess(line):
    text = clean_text(line)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences(text)
    data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return data

def classify_claim(claim, model):
    data = preprocess(claim)
    result = model.predict(data)
    return result[0]

def weighted_voting(weights):
    # Calculate votes for each label based on the weights
    false = 1*weights[0]
    partly = 1*weights[1]
    true = 1*weights[2]
    # Get the label with the most votes
    vote = max([false, partly, true])
    # Return the 'most popular' label
    if vote == false:
        return 0
    elif vote == partly:
        return 1
    else:
        return 2

def make_prediction(entry, model):
    # Classify based on claim text
    claim = entry['claim']
    result_claim = classify_claim(claim, model) # Returns a distribution
    # Get label from voting classifier
    prediction = weighted_voting(result_claim)
    return prediction