import os
import pickle
import numpy as np
from scipy import spatial
import sys
import collections

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Top 20 words for : a) first b) american c) would

==========================================================================
"""
'''Calculate embedding of word first'''
first_embedding = embeddings[dictionary['first']]
american_embedding = embeddings[dictionary['american']]
would_embedding = embeddings[dictionary['would']]
# print(first_embedding)
embed_word = {}
for key in dictionary.items():
    embed_word[key[0]] = embeddings[key[1]]


cosine_first = {}

for key in embed_word.items():
    vect1 = first_embedding
    vect2 = key[1]
    cosine_dist = 1 - spatial.distance.cosine(vect1, vect2)
    cosine_first[cosine_dist] = key[0]

cosine_first_sort = collections.OrderedDict(sorted(cosine_first.items(), reverse=True))
index = 1

'''Printing out closest '''
print("Top 20 words for \"first\": ")
for key in cosine_first_sort:
    if index == 1:
        index += 1
        continue
    if index > 21:
        break

    print(index-1,"->",cosine_first_sort[key])
    index += 1

'''Calculate embedding of word american'''
cosine_american = {}

for key in embed_word.items():
    vect1 = american_embedding
    vect2 = key[1]
    cosine_dist = 1 - spatial.distance.cosine(vect1, vect2)
    cosine_american[cosine_dist] = key[0]

cosine_american_sort = collections.OrderedDict(sorted(cosine_american.items(), reverse=True))
index = 1

print("\n\nTop 20 words for \"american\": ")
for key in cosine_american_sort:
    if index == 1:
        index += 1
        continue
    if index > 21:
        break

    print(index-1,"->",cosine_american_sort[key])
    index += 1


'''Calculate embedding of word would'''
cosine_would = {}

for key in embed_word.items():
    vect1 = would_embedding
    vect2 = key[1]
    cosine_dist = 1 - spatial.distance.cosine(vect1, vect2)
    cosine_would[cosine_dist] = key[0]

cosine_would_sort = collections.OrderedDict(sorted(cosine_would.items(), reverse=True))
index = 1

print("\n\nTop 20 words for \"would\": ")
for key in cosine_would_sort:
    if index == 1:
        index += 1
        continue
    if index > 21:
        break

    print(index-1,"->",cosine_would_sort[key])
    index += 1