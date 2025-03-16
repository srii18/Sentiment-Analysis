Amazon Review Sentiment Analysis
A machine learning project that analyzes Amazon product reviews to classify sentiments as positive or negative.
Overview
This project uses Natural Language Processing (NLP) techniques to analyze and classify Amazon product reviews. The sentiment analysis pipeline includes:

Text preprocessing and cleaning (removing stopwords)
Feature extraction using TF-IDF vectorization
Binary sentiment classification using Logistic Regression
Visualization of word frequencies and model performance metrics

Dataset
The project uses the "AmazonReview.csv" dataset containing product reviews with their corresponding sentiment ratings (1-5 stars). The sentiment ratings are converted to binary classes:

Ratings 1-3: Negative (0)
Ratings 4-5: Positive (1)

Features

Data Cleaning: Removes missing values and stopwords from reviews
Exploratory Data Analysis: Visualizes word frequency in negative reviews using word clouds
Text Vectorization: Converts text to numerical features using TF-IDF
Model Training: Implements Logistic Regression for sentiment classification
Performance Evaluation: Displays accuracy metrics and confusion matrix

Usage

Ensure you have the "AmazonReview.csv" file in the same directory as the script.
Run the script:
python sentiment_analysis.py