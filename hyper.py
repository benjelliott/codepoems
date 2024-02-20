from array import ArrayType
import os, glob
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import pandas as pd
import numpy as np
import scipy.stats as stats
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gensim.downloader

# this script is an implementation of a clustering algorithm called fast search of density peaks
# used to tune the hyper parameters of an LDA model for the generation of poetry

# variables

poems = []
n_features = 150
folder_path = '/Users/benelliott/Desktop/code poems/poems'

# stopwords

stopwords = stopwords.words('english')

extra = ["’", 'like', 'let', 've', 'would', 'saw', 'still', 'seen', 'see', 'could', 'say', 'wa', 'make', 'around', 'much', 'feel', 'go', 'got', 'sometimes', 'back', 'said', 'come']
for word in extra:
    stopwords.append(word)

# reading

for filename in glob.glob(os.path.join(folder_path, '*.txt')):
    with open(filename, 'r') as f:
        poems.append(f.read())

# lemmatizing

poemclean = []
poemblob = []
for poem in poems:
    blob = TextBlob(poem)
    poemblob.append([str.lower(w.lemmatize()) for w in blob.words if str.lower(w.lemmatize()) not in stopwords])
    poemclean.append(" ".join([w.lemmatize() for w in blob.words]))

# word2vec

w2v = Word2Vec(sentences=poemblob, vector_size=100, window=5, min_count=1)
words = w2v.wv.index_to_key
vecs = w2v.wv
del w2v

def dij(vecs):
    distances = []
    for i in range(0, len(words)):
        temp = []
        for j in range(0, len(words)):
            temp.append(vecs.distance(words[i], words[j]))
        distances.append(temp)
    
    return distances

def localdensity(distances, min_dist):
    density = []
    for i in range(0, len(words)):
        temp = []
        for j in range(0, len(words)):
            temp.append((distances[i][j] - min_dist) < 0)
        density.append(np.sum(temp))
    return density

def delta(distances, density):
    out = []
    for i in range(0, len(words)):
        temp = []
        for k in range(0, len(words)):
            if density[i] < density[k]:
                temp.append(distances[i][k])
        if len(temp) > 0:
            out.append(min(temp))
        else:
            out.append(max(distances[i]))
    
    return out

dijs = dij(vecs)
densities = localdensity(dijs, 0.78)
deltas = delta(dijs, densities)

from matplotlib.patches import Polygon
rectangle = plt.Rectangle((30, 0.9), 5, 0.5, alpha=0.5, color='green')
plt.scatter(densities, deltas)
plt.gca().add_patch(rectangle)
plt.xlabel(r"$\rho$")
plt.ylabel(r"$\delta$")
plt.title("Distance to Nearest High Density Neighbor vs Local Density")
plt.savefig("wiki choice of topic hyperparameter")
