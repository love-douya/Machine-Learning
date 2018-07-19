import spacy

# SpaCy Pipeline and Properties
# load the default model which is english-core-web.
nlp = spacy.load('en_core_web_sm')
#type(nlp)
 
document = open('Tripadvisor_hotelreviews_Shivambansal.txt', 'r', encoding = "UTF-8").read()
document = nlp(document)


# print(dir(document))

# Tokenization

print(document[0])

# last token of the doc
print(document[len(document) - 5])

# get all tags
all_tags = {w.pos: w.pos_ for w in document}
print(all_tags)

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