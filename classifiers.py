import random
from statistics import mode

TRUE = 2
PARTLY = 1
FALSE = 0

def classify_uniform_random(cl): return random.randint(0, 2)
def classify_all_true(cl): return TRUE
def classify_all_partly(cl): return PARTLY
def classify_all_false(cl): return FALSE

def classify_weighted_random(cl):
    result = [FALSE] * 50 + [PARTLY] * 40 + [TRUE] * 10
    return random.choice(result)

# false:  {'min': 15, 'max': 7251, 'mean': 142.4341252699784, 'median': 101.0, 'pstdev': 252.27116040305341}
# partly: {'min': 22, 'max': 8441, 'mean': 140.84715547977058, 'median': 111, 'pstdev': 196.08532387439064}
# true:   {'min': 20, 'max': 5716, 'mean': 124.96403301886792, 'median': 104.0, 'pstdev': 172.86696816567172}
def classify_claim_len(cl):
    size = len(cl['claim'])
    if size >= (142.34 + 140.85) / 2:
        return TRUE
    elif size >= (1440.85 + 124.96) / 2:
        return PARTLY
    else:
        return FALSE

# false:  {'min': 2, 'max': 66, 'mean': 5.262688984881209, 'median': 4.0, 'pstdev': 4.43623540426079}
# partly: {'min': 2, 'max': 41, 'mean': 4.841574949620214, 'median': 4, 'pstdev': 3.2703837621469978}
# true:   {'min': 2, 'max': 27, 'mean': 4.399174528301887, 'median': 3.0, 'pstdev': 3.0169293207073387}
def classify_related_count(cl):
    size = len(cl['related_articles'])
    if size >= (5.26 + 4.84) / 2:
        return TRUE
    elif size >= (4.84 + 4.40) / 2:
        return PARTLY
    else:
        return FALSE

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
        return TRUE
    elif size >= (4.84 + 4.40) / 2:
        return PARTLY
    else:
        return FALSE

def voting_classifier(claim, classifiers):
    votes = []
    result = None
    # Each classifier returns a 'vote' for the claim's label
    votes = [clf(claim) for clf in classifiers]
    # Pick the most frequently voted label
    try:
        result = mode(votes)
    # Or in the event of a tie, pick the smallest label value
    except:
        result = min(votes)
    return result