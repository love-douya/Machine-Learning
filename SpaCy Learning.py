import spacy

# SpaCy Pipeline and Properties
# load the default model which is english-core-web.
nlp = spacy.load('en_core_web_sm')
#type(nlp)
 
document = open('Tripadvisor_hotelreviews_Shivambansal.txt', 'r', encoding = "UTF-8").read()
document = nlp(document)


# print(dir(document))

# Tokenization

# print(document[0])

# last token of the doc
# print(document[len(document) - 5])

# get all tags
all_tags = {w.pos: w.pos_ for w in document}
# print(all_tags)

# define some parameters
noisy_pos_tags = ['PROP']
min_token_length = 2

# Function to check if the token is a noise or not
def isNoise(token):
    is_noise = False
    if token.pos_  in noisy_pos_tags:
        is_noise = True
    elif token.is_stop == True:
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    return is_noise

def cleanup(token, lower = True):
    if lower:
        token = token.lower()
    return token.strip()

# top unigrams used in the reviews
from collections import Counter
cleaned_list = [cleanup(word.string) for word in document if not isNoise(word)]
# print(Counter(cleaned_list).most_common(5))

# Entity Detection
labels = set([w.label_ for w in document.ents])
for label in labels:
    entities = [cleanup(e.string, lower = False) for e in document.ents if label == e.label_]
    entities = list(set(entities))
    #print(label, entities)

# Dependency Parsing
'''
One of the most powerful feature of spacy is the extremely fast and accurate syntactic dependency parser which can be accessed via lightweight API. 
The parser can also be used for sentence boundary detection and phrase chunking. 
The relations can be accessed by the properties “.children” , “.root”, “.ancestor” etc.
'''

# extract all review sentences that contains the term - hotel
hotel = [sent for sent in document.sents if 'hotel' in sent.string.lower()]

# Create dependency tree
sentence = hotel[2] 
for word in sentence:
    pass
    #print(word, ': ', str(list(word.children)))

# check all adjectives used with a word
def pos_words(sentence, token, ptag):
    sentences = [sent for sent in sentence.sents if token in sent.string]
    pwrds = []
    for sent in sentences:
        for word in sent:
            if token in word.string:
                pwrds.extend([child.string.strip() for child in word.children if child.pos_ == ptag])
    return Counter(pwrds).most_common(10)

#print(pos_words(document, 'hotel', 'ADJ'))

# Noun Phrases
# Use dependancy trees to generate noun phrases

doc = nlp('I love data science on analytics vidhya')
for np in doc.noun_chunks:
    pass
    #print(np.text, np.root.dep_, np.root.head.text)

# Word to Vectors Integration
'''
Spacy also provides inbuilt integration of dense, real valued vectors representing distributional similarity information. 
It uses GloVe vectors to generate vectors. 
GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
'''

from numpy import dot
from numpy.linalg import norm
from spacy.lang.en import English

parser = English()

# Generate word vector of the word - apple
apple = parser.vocab['apple']
# Cosine similarity function
cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
others = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != "apple"})

# sort by similarity score
others.sort(key = lambda w: cosine(w.vector, apple.vector))
others.reverse()

print("top most similar words to apple:")
for word in others[:10]:
    print(word.orth_)

