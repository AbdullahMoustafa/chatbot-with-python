# chatbot-with-python
Chatbot GUI - Deep learning Model


## Retrieval based Chatbots
A retrieval-based chatbot uses predefined input patterns and responses.
It then uses some type of heuristic approach to select the appropriate response.
It is widely used in the industry to make goal-oriented chatbots where we can customize the tone and flow of the chatbot to drive our customers with the best experience.


This project with source code, Shows how to build a chatbot using deep learning techniques. The chatbot will be trained on the dataset which contains categories (intents) of json file, pattern and responses. We use a special recurrent neural network (LSTM) to classify which category the user’s message belongs to and then we will give a random response from the list of responses.
based on NLTK, Keras, Python, and tensorflow.

## The Dataset
The dataset we will be using is ‘intents.json’. This is a JSON file that contains the patterns we need to find and the responses we want to return to the user.
NOTE: The intents.json file is changable you can change its content depends on your application and then retrain the model.

## Prerequisites
pip install tensorflow, keras, pickle, nltk

