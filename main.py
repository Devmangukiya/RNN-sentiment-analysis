## Import Libraries and load the model.
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from my_custom_layers import CustomLayer


## Load the model IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

## Load the pre-tranied model with ReLU activation
custom_objects = {'CustomLayer': CustomLayer}
model = load_model('simple_rnn_imdb.h5',custom_objects=custom_objects)


## Helper function
# Function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## Design Streamlit app.
import streamlit as st
st.title('IMDB movie review sentiment analysis')
st.write('Enter a movie review to classify it as positive or negative')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')