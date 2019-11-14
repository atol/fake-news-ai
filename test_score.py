import json
import glob
from classifiers import *

ARTICLES_FILEPATH = 'dataset/articles'

# F1 score for predictions.txt
def f1(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('output/test.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = f1_score(true, pred, average='macro')
    return result

# Matthews correlation coefficient score for predictions.txt
def mcc(claims):
    true = [cl['label'] for cl in claims]
    
    pred = []
    f = open('output/test.txt', 'r')
    for l in f:
        line = l.strip()
        pred.append(int(line[-1:]))

    result = matthews_corrcoef(true, pred)
    return result

def get_articles():
    articles = []

    # Load all articles from directory
    for file in glob.glob(os.path.join(ARTICLES_FILEPATH, '*.txt')):
        with open(file) as f:
            body = " ".join(line for line in f)
        
        base = os.path.basename(file)
        file_name = os.path.splitext(base)[0]
        
        article = (os.path.basename(file_name), body)
        articles.append(article)

    # Convert articles to dataframe
    articles_df = pd.DataFrame(articles)
    articles_df.columns = ['article_id', 'article']

    # Convert dataframe to dictionary
    article_dict = articles_df.set_index('article_id')['article'].to_dict()

    return article_dict

if __name__ == '__main__':
    articles = get_articles()

    # Set up classifiers and weights
    classifiers = get_classifiers()
    weights = get_weights(classifiers, eval_acc, articles)
    
    for i in range(len(classifiers)):
        print(classifiers[i].__name__)
        print('weight:', weights[i])
        print()

    # Read in the metadata file
    with open("input/test.json", 'r') as f:
        claims = json.load(f)
    
    # Create a predictions file
    print('Writing predictions to:', "output/test.txt")
    with open("output/test.txt", 'w') as f:
        for claim in claims:
            f.write('%d,%d\n' % (claim['id'], weighted_voting_classifier(claim, classifiers, weights, articles)))
    print('Finished writing predictions.')
    
    # Compute results
    print("\nF1 score:", f1(claims))