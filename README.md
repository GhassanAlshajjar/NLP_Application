NLP Application - Data Analyzer

Overview

This repository contains a Natural Language Processing (NLP) application built using Python's Tkinter framework for GUI development. The app provides an intuitive interface for data analysis, text preprocessing, visualization, and machine learning model application.

With the ability to handle CSV/JSON datasets, the app simplifies NLP workflows, making it suitable for both beginners and experts.

Features

1. Dataset Management

    Upload datasets in CSV or JSON format.
    
    View datasets with a scrollable and paginated table.
    
    Display metadata such as number of rows, columns, and dataset size.

2. Text Preprocessing

    Remove nulls, convert text to lowercase, and strip punctuation.
    
    Remove stopwords, and apply stemming or lemmatization.
    
    Cleaned data can be previewed and exported.

3. Visualization

    Generate visualizations like:
    
      Bar plots, pie charts, and word clouds.
      
      Word frequency and text length distributions.
      
      Line charts for time-series data.
      
      Customize chart inputs such as categorical columns and metrics.

4. Text Representation

    Support for advanced vectorization techniques:
    
      TF-IDF: Identify term importance.
      
      Bag of Words (BoW): Frequency of words.
      
      Named Entity Recognition (NER): Extract entities.
      
      N-Grams: Frequent word combinations.
      
      Word2Vec: Semantic word embeddings.
      
      View vectorized matrices with scrollable tables.

5. Model Application

    Train machine learning models with options like:
    
      Naive Bayes, Logistic Regression, Decision Trees, SVM, KNN.
      
      Evaluate models with metrics including:
      
      Accuracy, precision, recall, and F1-score.
      
      Confusion matrix visualization.
      
      Split datasets into training and test sets with customizable ratios.




Getting Started

  Prerequisites

  Ensure the following are installed on your machine:

  Python 3.8+

  Required libraries:

  pip install pandas numpy nltk spacy matplotlib seaborn wordcloud scikit-learn gensim


Download NLTK corpora:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

Install the Spacy English model:

python -m spacy download en_core_web_sm

Running the Application

Clone the repository:

git clone <repository-url>
cd <repository-name>

Run the Python script:

python main.py



How to Use


1. Upload Dataset

Click Upload Dataset in the "Dataset View" tab.

Select a CSV/JSON file to load data.

2. Preprocess Data

Navigate to the "Text Pre-Processing" tab.

Select preprocessing options and generate cleaned data.

Export cleaned data if needed.

3. Visualize Data

Use the "Visualization" tab to generate visualizations.

Select categorical columns, metrics, and chart types.

4. Represent Text

In the "Text Representation" tab, choose a vectorization method.

View vectorized matrices and corresponding charts.

5. Train Models

Go to the "Model Application" tab.

Select feature and label columns.

Choose a machine learning model and apply it.



Project Structure

-> main.py: Main script for the application.



