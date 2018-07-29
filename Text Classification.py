# We need Pandas Scikit-learn XGBoost TextBlob keras

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, XGBoost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

# load the dataset

data = open('../amazon_review_full_csv/train.csv').read()
data = open()