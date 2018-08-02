# We need Pandas Scikit-learn XGBoost TextBlob keras

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, XGBoost, numpy, textblob, string
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

# Feature Engineering

# Count Vectors as features
count_vect = CountVectorizer(analyzer = 'word', token_pattern = r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)