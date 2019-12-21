import os
import torch
import numpy as np
import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


j_l = []
with open('./joke_text.txt', 'r') as f:
    line = f.readline()
    while line:
        j_l.append(line)
        line = f.readline()
# print(len(j_l))


tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),
                             tags=[str(i)]) for i, _d in enumerate(j_l)]


max_epochs = 200
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
# print("Model Saved")

model= Doc2Vec.load("d2v.model")

def cal_rating(X, i, j):
    sum1 = 0
    sum2 = 0
    for k in range(len(X[0])):
        if j != k and mask[i][k]:
            sim = model.docvecs.n_similarity(str(j), str(k))
            sum1 += sim * X[i][k] 
            sum2 += sim
    return sum1 / sum2

def cal_err(X, n):
    n_users = len(X)
    n_items = len(X[0])
#     Y = np.zeros(shape=(n_users,n_items), dtype=np.float32)
    err = 0
    cnt = 0
    for i in range(n):
        for j in range(n_items):
            if mask[i][j] != 0:
                cnt += 1
                err += abs(cal_rating(X, i, j) - X[i][j])
    return err / cnt

print('err=', cal_err(X, 500))

