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

## File Structure
Intents.json – The data file which has predefined patterns and responses.
train_chatbot.py – In this Python file, A script to build the model and train our chatbot.
Words.pkl – This is a pickle file in which I store the words Python object that contains a list of our vocabulary.
Classes.pkl – The classes pickle file contains the list of categories.
Chatbot_model.h5 – This is the trained model that contains information about the model and has weights of the neurons.
Chatgui.py – This is the Python script in which I implemented GUI for our chatbot. Users can easily interact with the bot.

## Steps
1. Import and load the data file
2. Preprocess data
3. Create training and testing data
4. Build the model
5. Predict the response

## 1. Import and load the data file
First, make a file name as train_chatbot.py. We import the necessary packages for our chatbot and initialize the variables we will use in our Python project.
The data file is in JSON format so we used the json package to parse the JSON file into Python.


## 2.  Preprocess data
When working with text data, we need to perform various preprocessing on the data before we make a machine learning or a deep learning model. Tokenizing is the most basic and first thing you can do on text data. Tokenizing is the process of breaking the whole text into small parts like words.
Than we iterate through the patterns and tokenize the sentence using nltk.word_tokenize() function and append each word in the words list. We also create a list of classes for our tags.
lemmatize each word and remove duplicate words from the list. Lemmatizing is the process of converting a word into its lemma form and then creating a pickle file to store the Python objects which we will use while predicting.

## 3. Create training and testing data
Now, we will create the training data in which we will provide the input and the output. Our input will be the pattern and output will be the class our input pattern belongs to. But the computer doesn’t understand text so we will convert text into numbers.

## 4. Build the model
We have our training data ready, now we will build a deep neural network that has 3 layers. We use the Keras sequential API for this. After training the model for 200 epochs, we achieved 100% accuracy on our model. Let us save the model as ‘chatbot_model.h5’.

## 5. Predict the response (Graphical User Interface)
Now to predict the sentences and get a response from the user to let us create a new file ‘chatapp.py’.
We will load the trained model and then use a graphical user interface that will predict the response from the bot. The model will only tell us the class it belongs to, so we will implement some functions which will identify the class and then retrieve us a random response from the list of responses.
Again we import the necessary packages and load the ‘words.pkl’ and ‘classes.pkl’ pickle files which we have created when we trained our model.

To predict the class, we will need to provide input in the same way as we did while training. So we will create some functions that will perform text preprocessing and then predict the class.

After predicting the class, we will get a random response from the list of intents.

we will code a graphical user interface. For this, we use the Tkinter library which already comes in python. We will take the input message from the user and then use the helper functions we have created to get the response from the bot and display it on the GUI. Here is the full source code for the GUI.

##6. Run the chatbot
To run the chatbot, we have two main files; train_chatbot.py and chatapp.py.
First, we train the model using the command in the terminal:
python train_chatbot.py

If we don’t see any error during training, we have successfully created the model. Then to run the app, we run the second file.

python chatgui.py

