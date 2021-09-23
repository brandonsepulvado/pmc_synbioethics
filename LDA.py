# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 07:54:11 2021
@author: ddabl

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import io
import os.path
import re
import tarfile
import smart_open
import nltk
nltk.download('wordnet')
# Tokenize the documents.
from nltk.tokenize import RegexpTokenizer
# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer
# Remove rare and common tokens.
from gensim.corpora import Dictionary
# Train LDA model.
from gensim.models import LdaModel
from gensim.models import Phrases
import pandas as pd

from stop_words import get_stop_words

##############################################################################
#Perform LDA to identify 5 classes through topic modeling
##############################################################################

#directory where data saved 
directory = '.../SBKS/data_0523/'

#subdirectories where json files located 
j_list = ['json_sb4_n']

file_list = []

count=0
for j in j_list:  
    dir1 = directory + j
    file = os.listdir(dir1)
    
    for f in file:
    #read file  
    
        file_address = directory + j + '/' + f
        #print(file_address)
        file_list.append(file_address)
        count+=1

print(len(file_list)) #10643

#all docs
docs_all = []
# Decided to initially represent a document based on its abstract

#docs that contain full abstract vs missing abstract
doc_ab = []

for d in file_list:
    
    temp_file = pd.read_json(d, lines=True)
    df_temp = pd.DataFrame(temp_file) 
   
    num_docs = 0

    #iterate through dataframe 
    for row, col in df_temp.iterrows():
        text = str(col['abstract']) #to avoid attribute error re:nan
        #print(text)
        docs_all.append(text)
        if text != '':
            doc_ab.append(text)
    
print(len(docs_all))
print(len(doc_ab)) 

############################################################################
#perform LDA
docs= doc_ab

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

#print(docs)

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

# create English stop words list
en_stop = get_stop_words('en')

docs = [[token for token in doc if not token in en_stop] for doc in docs]

"""
Use the WordNet lemmatizer from NLTK. A lemmatizer is preferred over a
 stemmer in this case because it produces more readable words. 
"""

lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

"""
Find bigrams in the documents. Bigrams are sets of two adjacent words.
 
"""
# Compute bigrams.
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

"""
Remove rare words and common words based on their document frequency. Below
 words that appear in less than 20 documents or in more than 30% of
 the documents are removed to aid with class overlapping. 
"""

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 
#20% of the documents (to isolate words that may appear in 5 topics 5 X 20%).

dictionary.filter_extremes(no_below=20, no_above=0.4)

"""
Finally, transform the documents to a vectorized form by computing the
 frequency of each word, including the bigrams.
"""

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

#number of tokens and documents we have to train on.

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

#Number of unique tokens: 6142
#Number of documents: 10058

"""
Training.  Selected 5 topics after some initial experimentation.
"""

# Set training parameters.
num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)


top_topics = model.top_topics(corpus) 

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

len(top_topics) #5


from pprint import pprint
pprint(top_topics)

len(top_topics[0][0])

for i in range(5):
    lentop = len(top_topics[i][0])
    print(lentop)
    topic_list=[]
    for n in range(lentop):
        topic_list.append(top_topics[i][0][n][1])
    print(topic_list)

