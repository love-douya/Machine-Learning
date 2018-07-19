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