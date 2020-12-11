#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:37:32 2019

@author: rakeshsoni
"""

import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize
from prettytable import PrettyTable


df = pd.read_csv("tennis_articles_v4.csv") #sample dataset
df.head()

lst=[] #list of final sentences
lsc=[]   #list of scores of final sentences
flsc=[]  #final scores(out of 1)

sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x]
sentences[:1]
#print(sentences)

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8') #https://nlp.stanford.edu/projects/glove/
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
  
sim_mat = np.zeros([len(sentences), len(sentences)])

from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
print(scores)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

for i in range(3):
   lst.insert(i,ranked_sentences[i][1])

print(lst)

for i in range(3):
   lsc.insert(i,ranked_sentences[i][0])

print(lsc)

for i in range(3):
    flsc.insert(i,(1/lsc[i]))

print(flsc)   
 
def Average(flsc): 
    return sum(flsc) / len(flsc)

avg=Average(flsc) 

def status(avg):
    if(avg<20.00):
        st='Red'
    elif(avg>20.01 and avg<50.99):
        st='Yellow'
    else:
        st='Green'
 
    return st

stf=status(avg)
        

t = PrettyTable(['Summary','Score','Status'])
t.add_row([lst,avg,stf])
print(t)


    
