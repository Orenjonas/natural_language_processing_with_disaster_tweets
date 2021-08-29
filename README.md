# Natural language processing with disaster tweets

This repo contains a jupyter notebook for analyzing and predicting disaster tweets, for the introductory
[Kaggle competition](https://www.kaggle.com/c/nlp-getting-started).

This notebook focuses on the data exploration and data cleaning.
The latest version of the notebook can be seen [here](https://htmlpreview.github.io/?https://github.com/Orenjonas/natural_language_processing_with_disaster_tweets/blob/main/nlp_with_disaster_tweets.html).

The [notebook](https://www.kaggle.com/orenjonas/glove-and-lstm-model) used for running the model can be found on my [kaggle profile](https://www.kaggle.com/orenjonas).

## Purpose
The purpose of this project is to learn the basics of implementing machine learning algorithms for natural
language processing, and catching up on the state of the art in the field.

I will create a local jupyter notebook to experiment with and implement different methods, for later
submission to the kaggle website when the predictions are satisfactory, and the notebook is readable and concise.

## Things to try
- Use a docker image
- Use "location" and "keyword" in analysis?
- Fine tune LSTM model
    - Inspect incorrectly classified tweets
    - Context specific embeddings
    - Add spelling corrections
    - Play with training parameters
    - Add estimated sentiment of tweet to the data
- Compare with a RoBERTa model
