# Fake News AI

A fake news detector developed as part of the Leaders Prize: Fact or Fake News? competition and SFU Special Research Project (CMPT 415) 
course. The goal was to implement an AI algorithm that correctly labels a given claim as `true`, `partly true` or `false`.

The program takes as input a metadata file with information about a series of claims and an empty `label` field for each claim. 
The output is a predictions file with a rating of `true`, `partly true` or `false` in the `label` field for each claim.

## Dataset

The dataset consists of claims and associated metadata downloaded from 9 fact-checking websites: politifact.com, snopes.com,
washingtonpost.com, weeklystandard.com, africacheck.org, factscan.ca, factcheck.afp.com, polygraph.info, factcheck.org.

An example of a claim is shown below:
```
"claim": "\"We have $91 billion going to Puerto Rico. We have $29 billion to Texas and $12 billion to Florida for the hurricanes.\"",
"claimant": "Donald Trump",
"date": "2019-03-28", 
"label": 1, 
"related_articles": [60922, 41290, 28742],
"id": 123
```
where `claim` is the statement to be evaluated on truthfulness, `claimant` is the entity who made the claim, `date` is when 
the claim was made, `label` is the truth rating of the claim (0: false, 1: partly true, 2: true), `related_articles` is a list of
article IDs that point to the names of text files containing the articles, and `id` is the unique identifier for each claim.

## Evaluation

Submission scores were calculated using the macro average F1 score of the outputted truth ratings. The formula is defined as follows:

![score = \frac{2*P*R}{P+R}](https://render.githubusercontent.com/render/math?math=score%20%3D%20%5Cfrac%7B2*P*R%7D%7BP%2BR%7D)

where P is precision and R is recall, defined as:

![P = \frac{P_{true}+P_{partly}+P_{false}}{3}](https://render.githubusercontent.com/render/math?math=P%20%3D%20%5Cfrac%7BP_%7Btrue%7D%2BP_%7Bpartly%7D%2BP_%7Bfalse%7D%7D%7B3%7D)

![R = \frac{R_{true}+R_{partly}+R_{false}}{3}](https://render.githubusercontent.com/render/math?math=R%20%3D%20%5Cfrac%7BR_%7Btrue%7D%2BR_%7Bpartly%7D%2BR_%7Bfalse%7D%7D%7B3%7D)

The precision and recall for each class are defined as:

![P_{class} = \frac{TP_{class}}{TP_{class}+FP_{class}}](https://render.githubusercontent.com/render/math?math=P_%7Bclass%7D%20%3D%20%5Cfrac%7BTP_%7Bclass%7D%7D%7BTP_%7Bclass%7D%2BFP_%7Bclass%7D%7D)

![R_{class} = \frac{TP_{class}}{TP_{class}+FN_{class}}](https://render.githubusercontent.com/render/math?math=R_%7Bclass%7D%20%3D%20%5Cfrac%7BTP_%7Bclass%7D%7D%7BTP_%7Bclass%7D%2BFN_%7Bclass%7D%7D)

where TP is the number of true positives, FP is the number of false positives, and FN is the number of false negatives.

## Results

| Classifier	                            | Validation Score | Test Score |
| --------------------------------------- |:----------------:| ----------:|
| Uniform Random Guesser                  | 0.309249	       | 0.250182   |
| Weighted Random Guesser                 |	0.335557	       | 0.251016   |
| Claim Length	                          | 0.203872	       | n/a        |
| Word Count	                            | 0.217368	       | n/a        |
| Related Article Count	                  | 0.258209	       | n/a        |
| Related Article ID	                    | 0.345475         | 0.26116    |
| Claimant	                              | 0.438962	       | 0.382788   |
| Ensemble	                              | 0.376860         | 0.293169   |
| Naive Bayes (Claim)	                    | 0.441848	       | 0.376152   |
| Naive Bayes (Claim) + SMOTE	            | 0.482802         | 0.428168   |
| Naive Bayes (Claimant) + SMOTE	        | 0.440491         | n/a        |
| Naive Bayes (Claim + Claimant) + SMOTE	| 0.478622         | 0.43149    |
| CNN	                                    | 0.605	           | 0.409578   |
| **CNN + LIAR**                          | **0.6798**       | **0.435576** |