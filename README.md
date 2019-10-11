# fake-news-ai
Leaders Prize: Fact or Fake News?

## Phase 1 – Truth Rating

In Phase 1, each team will create an algorithm that fact-checks claims and provides a truth rating for each. Claims shall be rated as TRUE, PARTLY TRUE or FALSE.

Algorithms should take as input, a metadata file with information about a series of claims. This file will be in the same format as the training set of metadata, except with the TRUTH RATING field empty – this is what your algorithm will need to populate.

Your team's algorithm will then output a metadata file with a truth rating for each of the claims.

From June 19, 2019 to November 18, 2019 **participants may submit their algorithm a maximum of once per day** to be evaluated using the validation set. Scores will appear on the Leaderboard. The Leaderboard will be updated once per day. The validation set will be updated with additional data on August 1, September 1, October 1 and November 1 to help competitors create more robust solutions.

**By November 18, 2019, participants may designate one of their existing submissions as the submission to be used for the evaluation against the test set.** If no submission is specifically designated, the most recent submission will be used for Phase 1 evaluation.

Solutions will be evaluated between November 19, 2019 and December 18, 2019 by scoring submissions against the test data set. On December 19, 2019 Phase 1 scores will be released and the 10 teams with the highest score on the test set will be invited to continue with Phase 2 of the Leaders Prize.

## Submission Instructions

A submission consists of a Docker container.

To build the container, run `` docker build -t image_name . `` in the same directory as the Dockerfile.

Then, to test running the submission locally, run this command:

```
docker run \
  -v $DATASET_PATH:/usr/local/dataset/:ro \
  -v $OUTPUT_PATH:/usr/local/ \
  --name container_name \
  image_name
```

where ``$DATASET_PATH`` is the folder with the dataset on your local machine and ``$OUTPUT_PATH`` is where you want to save the
predictions file.

To submit your Docker container to the leaderboard, you will first need to save it as a tar file using this command:

``docker save -o tar_file_name.tar image_name``
