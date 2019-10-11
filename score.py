# Simple scoring function

if __name__ == '__main__':
    num_claims = 15555

    with open ('predictions.txt', 'r') as predictions:
        with open('solutions.txt', 'r') as solutions:
            same = set(predictions).intersection(solutions)
    
    print("score: {:.5f}".format(len(same)/num_claims))