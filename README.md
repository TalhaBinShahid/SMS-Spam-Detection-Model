# SMS Spam Classifier

This project is an SMS Spam Classifier that predicts whether a given SMS or email message is spam or not. The model was built using Python, leveraging natural language processing techniques and machine learning algorithms. Additionally, a web application was developed using Streamlit to provide an interactive interface for users.

## Table of Contents
- Overview
- Dataset
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Building
- Web Application
- Results
- Dependencies
- Conclusion

## Overview
This project involves classifying SMS and email messages as spam or ham (not spam) using natural language processing (NLP) and machine learning techniques. A web application was also developed using Streamlit to make the model accessible for real-time predictions.

## Dataset
The dataset used for this project was sourced from the UCI Machine Learning Repository. It contains a collection of SMS messages labeled as "spam" or "ham" (not spam).

## Data Cleaning
Several preprocessing steps were performed to clean the data:
- Dropped unnecessary columns.
- Renamed columns for better understanding.
- Removed duplicate entries.

## Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to understand the distribution of features such as the number of characters, words, and sentences in the messages. Visualizations include histograms and heatmaps to reveal correlations.

## Data Preprocessing
Text data was preprocessed through the following steps:
- Lowercasing the text.
- Tokenization.
- Removal of special characters and stop words.
- Stemming.

## Model Building
Three different models were built using the following machine learning algorithms:
1. Gaussian Naive Bayes
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes

The models were evaluated using metrics such as accuracy, confusion matrix, and precision score. The Multinomial Naive Bayes model was chosen for its superior performance.

## Web Application
A web application was developed using Streamlit to provide a user-friendly interface for predicting whether a message is spam or not. The application includes:
- A text input field where users can enter an SMS or email message.
- A "Predict" button that triggers the prediction process.
- A display area that shows whether the message is classified as "Spam" or "Not Spam."

### How the Web App Works:
1. The input text is preprocessed (lowercased, tokenized, stemmed, and cleaned).
2. The preprocessed text is vectorized using the pre-trained TF-IDF vectorizer.
3. The vectorized text is passed through the trained model to make a prediction.
4. The result is displayed on the web interface.

## Results
The Multinomial Naive Bayes model achieved the best performance with the following metrics:
- Accuracy: 97.1%
- Precision: 100.0%

## Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- nltk
- streamlit

## Conclusion
The SMS Spam Classifier successfully predicts whether a message is spam or ham, and the Streamlit web application provides a convenient way for users to make real-time predictions. Future improvements could include using more advanced models like deep learning techniques for even better accuracy.
