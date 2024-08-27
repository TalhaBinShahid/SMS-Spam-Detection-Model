# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))
#     return " ".join(y)


# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# st.title("SMS/Email Spam Detector")
# input_sms = st.text_input("Enter the message")

# if st.button('Predict'):
#     #Preprocessing
#     transformed_sms = transform_text(input_sms)

#     # Vectorize
#     vector_input = tfidf.transform([transformed_sms])

#     # Predict
#     result = model.predict(vector_input)[0]

#     # Display
#     if result==1:
#         st.header('Spam')
#     else:
#         st.header("Not Spam")

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english')]
    
    # Stem the words
    y = [ps.stem(i) for i in y]
    
    # Return the transformed text
    return " ".join(y)

# Load the pre-trained TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title
st.title("SMS/Email Spam Detector")

# Text input for the message
input_sms = st.text_area("Enter the message")

# Prediction button
if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # Vectorize the transformed message
    vector_input = tfidf.transform([transformed_sms])

    # Predict using the loaded model
    result = model.predict(vector_input)[0]

    # Display the prediction result
    if result == 1:
        st.header('Spam')
    else:
        st.header("Not Spam")
