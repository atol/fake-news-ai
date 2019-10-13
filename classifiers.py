import random
import pickle
from statistics import mode

claimants = pickle.load( open( "preprocessing/labelled_claimants.p", "rb" ) )

def classify_uniform_random(cl): return random.randint(0, 2)
def classify_all_true(cl): return 2
def classify_all_partly(cl): return 1
def classify_all_false(cl): return 0

def classify_weighted_random(cl):
    result = [0] * 50 + [1] * 40 + [2] * 10
    return random.choice(result)

# false:  {'min': 15, 'max': 7251, 'mean': 142.4341252699784, 'median': 101.0, 'pstdev': 252.27116040305341}
# partly: {'min': 22, 'max': 8441, 'mean': 140.84715547977058, 'median': 111, 'pstdev': 196.08532387439064}
# true:   {'min': 20, 'max': 5716, 'mean': 124.96403301886792, 'median': 104.0, 'pstdev': 172.86696816567172}
def classify_claim_len(cl):
    size = len(cl['claim'])
    if size >= (142.34 + 140.85) / 2:
        return 2
    elif size >= (1440.85 + 124.96) / 2:
        return 1
    else:
        return 0

# false:  {'min': 2, 'max': 66, 'mean': 5.262688984881209, 'median': 4.0, 'pstdev': 4.43623540426079}
# partly: {'min': 2, 'max': 41, 'mean': 4.841574949620214, 'median': 4, 'pstdev': 3.2703837621469978}
# true:   {'min': 2, 'max': 27, 'mean': 4.399174528301887, 'median': 3.0, 'pstdev': 3.0169293207073387}
def classify_related_count(cl):
    size = len(cl['related_articles'])
    if size >= (5.26 + 4.84) / 2:
        return 2
    elif size >= (4.84 + 4.40) / 2:
        return 1
    else:
        return 0

# Total # of claims:
#    0=7408, 1=6451, 2=1696
#
# Avg # words per claim:
#    0=23.4, 1=23.1, 2=20.6
#
# Avg word length per claim:
#    0=5.2, 1=5.2, 2=5.2
#
def classify_word_count(cl):
    size = len(cl['claim'].split())
    if size >= (5.26 + 4.84) / 2:
        return 2
    elif size >= (4.84 + 4.40) / 2:
        return 1
    else:
        return 0

def classify_claimant(cl):
    if cl['claimant'] in claimants:
        return claimants[cl['claimant']]
    else:
        return 0

def voting_classifier(claim, classifiers):
    # Each classifier returns a 'vote' for the claim's label
    votes = [clf(claim) for clf in classifiers]
    # Pick the most frequently voted label
    try:
        result = mode(votes)
    # Or in the event of a tie, pick the smallest label value
    except:
        result = min(votes)
    return result