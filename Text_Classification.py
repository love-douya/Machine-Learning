# We need Pandas Scikit-learn XGBoost TextBlob keras

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# load the dataset

data = open('data/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split('\n')):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1])

# create a dataframe using texts and labels
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# Next, we will split the dataset into training and validation sets so that we can train and test classifier. 
# Also, we will encode our target column so that it can be used in machine learning models.

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# Feature Engineering

# Count Vectors as features
count_vect = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

# TF-IDF Vectors as features

#word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w{1, }', max_features = 5000)
tfidf_vect.fit(trainDF['text'])
