# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:42:11 2021

@author: Anny
"""


from gensim.models import KeyedVectors,word2vec,Word2Vec

import multiprocessing
import pandas as pd
data = pd.read_csv('preprocessed_dataset.txt')
X = data.iloc[:,4].values.astype('U')
for i in range(5545):
   sentences = list(X[i])
   model = Word2Vec(sentences, min_count=1, window=5,sg=0,workers=multiprocessing.cpu_count())
   model.save('word2vec.model')
   model = Word2Vec.load('word2vec.model')
   print(model)
   vec = model.wv.vectors 
   print(vec)
   
