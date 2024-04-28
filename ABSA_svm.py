#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import the required libaries
import xml.etree.ElementTree as ET  # Importing the ElementTree module for XML parsing
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing TfidfVectorizer for text vectorization
from sklearn.svm import SVC  # Importing Support Vector Classification from scikit-learn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Importing RandomForestClassifier and GradientBoostingClassifier
from sklearn.metrics import classification_report  # Importing classification_report for model evaluation
from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder for label encoding
import warnings
from sklearn.metrics import classification_report
# Filter out warnings related to undefined metrics
warnings.filterwarnings("ignore", category=UserWarning)


# Parse the XML file and extract review sentences with labels
def load_data_from_xml(xml_file):
    """
    Parses XML file to extract review sentences with labels.
    Args:
        xml_file (str): Path to the XML file.
    Returns:
        list: List of tuples containing review sentences with labels.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    reviews = []
    for review in root.findall('Review'):
        sentences = []
        for sentence in review.findall('sentences/sentence'):
            text = sentence.find('text').text
            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    category = opinion.get('category')
                    polarity = opinion.get('polarity')
                    target = opinion.get('target')
                    sentences.append((text, polarity, category, target))
        reviews.extend(sentences)
    return reviews

def load_predict_data_from_xml(xml_file):
    """
    Parses XML file to extract review sentences for prediction.
    Args:
        xml_file (str): Path to the XML file.
    Returns:
        list: List of tuples containing review sentences.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    reviews = []
    for review in root.findall('Review'):
        sentences = []
        for sentence in review.findall('sentences/sentence'):
            text = sentence.find('text').text
            sentences.append((text,))
        reviews.extend(sentences)
    return reviews

# Preprocess the data
def preprocess_data(data):
    """
    Preprocesses the data.
    Args:
        data (list): List of tuples containing data.
    Returns:
        tuple: Processed data.
    """
    X = [text for text, _, _, _ in data]
    y_polarity = [polarity for _, polarity, _, _ in data]
    y_category = [category for _, _, category, _ in data]
    y_target = [target for _, _, _, target in data]
    return X, y_polarity, y_category, y_target

def preprocess_predict_data(data):
    """
    Preprocesses the data for prediction.
    Args:
        data (list): List of tuples containing data.
    Returns:
        list: Preprocessed data.
    """
    X = [text for text, in data]
    return X

# Convert text data into numerical features
def vectorize_text(X_train, X_test):
    """
    Converts text data into numerical features using TF-IDF vectorization.
    Args:
        X_train (list): Training data.
        X_test (list): Testing data.
    Returns:
        tuple: Transformed training and testing data.
    """
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

# Train a machine learning model
def train_model(X_train, y_train):
    """
    Trains a machine learning model (SVM).
    Args:
        X_train (array): Training data.
        y_train (array): Training labels.
    Returns:
        object: Trained model.
    """
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates a machine learning model.
    Args:
        model (object): Trained model.
        X_test (array): Testing data.
        y_test (array): Testing labels.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Define a function to make predictions using the provided model
def predict(model, tfidf_vectorizer, new_sentences):
    """
    Makes predictions using the provided model.
    Args:
        model (object): Trained model.
        tfidf_vectorizer (object): Fitted TF-IDF vectorizer.
        new_sentences (list): New sentences to predict.
    Returns:
        array: Predicted labels.
    """
    # Ensure that the vectorizer is fitted
    if not hasattr(tfidf_vectorizer, 'vocabulary_'):
        raise ValueError("The TF-IDF vectorizer is not fitted")
    
    # Transform new sentences using the fitted vectorizer
    X_new_tfidf = tfidf_vectorizer.transform(new_sentences)
    
    # Predict using the provided model
    predictions = model.predict(X_new_tfidf)
    
    return predictions


if __name__ == "__main__":
    # Load training data
    train_data = load_data_from_xml("ABSA16_Restaurants_Train_SB1_v2.xml")
    X_train, y_polarity_train, y_category_train, y_target_train = preprocess_data(train_data)

    # Vectorize training data
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Train machine learning models
    model_polarity = train_model(X_train_tfidf, y_polarity_train)
    model_category = train_model(X_train_tfidf, y_category_train)
    model_target = train_model(X_train_tfidf, y_target_train)

    # Load testing data
    test_data = load_data_from_xml("EN_REST_SB1_TEST.xml.gold")
    X_test, y_polarity_test, y_category_test, y_target_test = preprocess_data(test_data)

    # Vectorize testing data
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Evaluate models on testing data
    print("Polarity Evaluation:")
    evaluate_model(model_polarity, X_test_tfidf, y_polarity_test)
    print("Category Evaluation:")
    evaluate_model(model_category, X_test_tfidf, y_category_test)
    #print("Target Evaluation:")
    #evaluate_model(model_target, X_test_tfidf, y_target_test)

    # Load data to predict
    predict_data = load_predict_data_from_xml("EN_REST_SB1_TEST.xml.A")
    X_predict = preprocess_predict_data(predict_data)

    # Vectorize data to predict
    X_predict_tfidf = tfidf_vectorizer.transform(X_predict)

    # Make predictions
    y_polarity_pred = predict(model_polarity, tfidf_vectorizer, X_predict)
    y_category_pred = predict(model_category, tfidf_vectorizer, X_predict)
    y_target_pred = predict(model_target, tfidf_vectorizer, X_predict)

    # Print predictions
    for sentence, polarity, category, target in zip(X_predict, y_polarity_pred, y_category_pred, y_target_pred):
        print("Sentence:", sentence)
        print("Predicted Polarity:", polarity)
        print("Predicted Category:", category)
        print()


# # Testing

# In[5]:


# Load data to predict
predict_data = load_predict_data_from_xml("EN_REST_SB1_TEST.xml.A")
X_predict = preprocess_predict_data(predict_data)


# Vectorize data to predict
X_predict_tfidf = tfidf_vectorizer.transform(X_predict)

# Make predictions
y_polarity_pred = predict(model_polarity, tfidf_vectorizer, X_predict)
y_category_pred = predict(model_category, tfidf_vectorizer, X_predict)
y_target_pred = predict(model_target, tfidf_vectorizer, X_predict)

# Print predictions
for sentence, polarity, category, target in zip(X_predict, y_polarity_pred, y_category_pred, y_target_pred):
    print("Sentence:", sentence)
    print("Predicted Polarity:", polarity)
    print("Predicted Category:", category)
    print("Predicted Target:", target)
    print()


# In[6]:


# Load data to predict
    
predict_data = "The food was tasting awful"
X_predict = [predict_data]

# Vectorize data to predict
X_predict_tfidf = tfidf_vectorizer.transform(X_predict)

# Make predictions
y_polarity_pred = predict(model_polarity, tfidf_vectorizer, X_predict)
y_category_pred = predict(model_category, tfidf_vectorizer, X_predict)
y_target_pred = predict(model_target, tfidf_vectorizer, X_predict)

# Print predictions
for sentence, polarity, category, target in zip(X_predict, y_polarity_pred, y_category_pred, y_target_pred):
    print("Sentence:", sentence)
    print("Predicted Polarity:", polarity)
    print("Predicted Category:", category)
    print("Predicted Target:", target)
    print()


# In[8]:


# Load data to predict
    
predict_data = "The waitress was neatly dressed and courteous!"
X_predict = [predict_data]

# Vectorize data to predict
X_predict_tfidf = tfidf_vectorizer.transform(X_predict)

# Make predictions
y_polarity_pred = predict(model_polarity, tfidf_vectorizer, X_predict)
y_category_pred = predict(model_category, tfidf_vectorizer, X_predict)
y_target_pred = predict(model_target, tfidf_vectorizer, X_predict)

# Print predictions
for sentence, polarity, category, target in zip(X_predict, y_polarity_pred, y_category_pred, y_target_pred):
    print("Sentence:", sentence)
    print("Predicted Polarity:", polarity)
    print("Predicted Category:", category)
    print("Predicted Target:", target)
    print()


# In[ ]:




