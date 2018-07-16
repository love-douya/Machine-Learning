# 2. Text Preprocessing

# 2.1 Noise Removal

noise_list = ["is", "a", "this", "..."]
def _remove_noise(input_text):
    words = input_text.split()
    noise_free_words = [word for word in words if word not in noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

#print(_remove_noise("this is a sample text"))

import re

regex_pattern = "#[\w]*"

def _remove_regex(input_text, regex_pattern):
    urls = re.finditer(regex_pattern, input_text)
    for i in urls:
        input_text = re.sub(i.group().strip(), '', input_text)
    return input_text

#print(_remove_regex("remove this #hashtag from analytics vidhya", regex_pattern))

# 2.2 Lexicon Normalization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "multiplying"
#print(lem.lemmatize(word, "v"))
#>> "multiply"
#print(stem.stem(word))
#>> "multipli"

# 2.3 Object Standardization

lookup_dict = {'rt': 'Retweet', 'dm': 'direct message', 'awsm': 'awespme', 'luv': 'love'}
def _lookup_words(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word)
    new_text = " ".join(new_words)
    return new_text

#print(_lookup_words("RT this is a retweeted tweeted tweet by Shivam Bansal"))

# 3.Text to Features (Feature Engineering on text data)

# 3.1 Syntactic Parsing

from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing on Analy tics Vidhya"
tokens = word_tokenize(text)
print(pos_tag(tokens))

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc_complete = [doc1, doc2, doc3]
doc_clean = [doc.split() for doc in doc_complete]

import gensim
from gensim import corpora

# Creating the term dictionary of our corpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim  library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix
ldamodel = Lda(doc_term_matrix, num_topics = 3, id2word = dictionary, passes = 50)

# Results
#print(ldamodel.print_topics())

# N-Grams as Features
"""
A combination of N words together are called N-Grams. 
N grams (N > 1) are generally more informative as compared to words (Unigrams) as features. 
Also, bigrams (N = 2) are considered as the most important features of all the others. 
The following code generates bigram of a text.
"""

def generate_ngrams(text, n):
    words = text.split()
    output = []
    for i in range(len(words) - n + 1):
        output.append(words[i: i + n])
    return output

#print(generate_ngrams('this is a sample text', 2))

# Term Frequency – Inverse Document Frequency (TF – IDF)

"""
Term Frequency (TF) – TF for a term “t” is defined as the count of a term “t” in a document “D”

Inverse Document Frequency (IDF) – IDF for a term is defined as logarithm of ratio of total documents available in the corpus and number of documents containing the term T.

TF . IDF – TF IDF formula gives the relative importance of a term in a corpus (list of documents), given by the following formula below. Following is the code using python’s scikit learn package to convert a text into tf idf vectors
"""

from sklearn.feature_extraction.text import TfidfVectorizer
obj = TfidfVectorizer()
corpus = ['This is sample document.', 'another random document.', 'third sample document text']
X = obj.fit_transform(corpus)
#print(X)

from gensim.models import Word2Vec
sentences = [['data', 'science'], ['vidhya', 'science', 'data', 'analytics'], ['machine', 'learning'], ['deep', 'learning']]

# train the model on your corpus
model = Word2Vec(sentences, min_count = 1)

print(model.similarity('data', 'science'))
print(model['learning'])

# Important tasks of NLP
# Text Classification

from textblob.classifiers import NaiveBayesClassifier as NBC 
from textblob import TextBlob

training_corpus = [('I am exhausted of this work.', 'Class_B'),
                   ("I can't cooperate with this", 'Class_B'),
                   ('He is my badest enemy!', 'Class_B'),
                   ('My management is poor.', 'Class_B'),
                   ('I love this burger.', 'Class_A'),
                   ('This is an brilliant place!', 'Class_A'),
                   ('I feel very good about these dates.', 'Class_A'),
                   ('This is my best work.', 'Class_A'),
                   ("What an awesome view", 'Class_A'),
                   ('I do not like this dish', 'Class_B')]

test_corpus = [ ("I am not feeling well today.", 'Class_B'), 
                ("I feel brilliant!", 'Class_A'), 
                ('Gary is a friend of mine.', 'Class_A'), 
                ("I can't believe I'm doing this.", 'Class_B'), 
                ('The date was good.', 'Class_A'), ('I do not enjoy my job', 'Class_B')]

model = NBC(training_corpus)
print(model.classify('Their code are amazing.'))
print(model.classify("I don't like their computer."))
print(model.accuracy(test_corpus))

# Scikit.Learn also provides a pipeline framework for text classification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm

# preparing data for SVM model (using the same training_corpus, text_corpus from naive bayes example)
train_data = []
train_labels = []
for row in training_corpus:
    train_data.append(row[0])
    train_labels.append(row[1])

test_data = []
test_labels = []
for row in test_corpus:
    test_data.append(row[0])
    test_labels.append(row[1])

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 4, max_df = 0.9)
# Train the feature vectors
train_vectors = vectorizer.fit_transform(train_data)
# Apply model on test data
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernal = linear
model = svm.SVC(kernel = 'linear')
model.fit(train_vectors, train_labels)
prediction = model.predict(test_vectors)

print(classification_report(test_labels, prediction))

# Text Matching / Similarity
# Levenshtein Distance

def levenshtein(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1 + 1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

print(levenshtein('analyze', 'analyse'))

# Phonetic Matching
'''
A Phonetic matching algorithm takes a keyword as input (person’s name, location name etc) and 
produces a character string that identifies a set of words that are (roughly) phonetically similar. 
It is very useful for searching large text corpuses, correcting spelling errors and matching relevant names. 
Soundex and Metaphone are two main phonetic algorithms used for this purpose. 
Python’s module Fuzzy is used to compute soundex strings for different words.
'''

