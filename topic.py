from array import ArrayType
import os, glob
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import numpy as np
import scipy.stats as stats
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# variables

poems = []
n_features = 150
n_components = 4
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

# tokenizing, cleaning stop words, and creating tf list

tf_vectorizer = CountVectorizer(max_features=n_features, stop_words=stopwords, ngram_range=(1, 1))
tf = tf_vectorizer.fit_transform(poemclean)

# word2vec

w2v = Word2Vec(sentences=poemblob, vector_size=100, window=5, min_count=1)
wv = w2v.wv
del w2v

# fitting model

lda = LatentDirichletAllocation(n_components=n_components, random_state=3).fit(tf, )

# getting topics and terms

topics = ["1", "2", "3", "4"]

def ldatopics(model, vectorizer):
    dist = []
    feature_names = vectorizer.get_feature_names_out()
    feature_probs = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    for i in range(n_components):
        array = []
        features_ind = feature_probs[i].argsort()
        features = [feature_names[i] for i in features_ind]
        array.append(features)
        feature_probs[i].sort()
        array.append(feature_probs[i])
        dist.append(array)

    return pd.DataFrame(dist)

# printing

distrib = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
fit_model = ldatopics(lda, tf_vectorizer)
for i in range(0, n_components):
    print(fit_model[0][i][-10:], fit_model[1][i][-10:])

# prompting

def prompt(n):
    out = []
    topic_dist = [1/n_components] * n_components
    for iter in range(n):
        topics = stats.multinomial(1, topic_dist)
        topic1 = sum(topics.rvs(1)[0]*range(0, n_components))
        sample1 = stats.multinomial(1, fit_model[1][topic1]).rvs(1)[0]
        index1 = sum(sample1 * range(n_features))
        word1 = fit_model[0][topic1][index1]
        topic_dist = []
        for i in range(n_components):
            if word1 in fit_model[0][i]:
                rank = fit_model[0][i].index(word1)
            else:
                rank = 0
            topic_dist.append(rank)
        topic_dist = topic_dist/np.sum(topic_dist)

        topics = stats.multinomial(1, topic_dist)
        topic2 = sum(topics.rvs(1)[0]*range(0, n_components))

        wordlist2 = []
        probs2 = []
        similar = np.array(wv.most_similar(word1, topn=n_features))
        for i in range(similar.shape[0]):
            if similar[i][0] in fit_model[0][topic2]:
                wordlist2.append(similar[i][0])
                # absolute value because turns are cool
                probs2.append(abs(float(similar[i][1])))
            else:
                pass
        probs2 = np.array(probs2)/sum(probs2)
        sample2 = stats.multinomial(1, probs2).rvs(1)[0]
        index2 = sum(sample2 * range(len(probs2)))
        word2 = wordlist2[index2]
        
        topic_dist = []
        for i in range(n_components):
            if word2 in fit_model[0][i]:
                rank = fit_model[0][i].index(word1)
            else:
                rank = 0
            topic_dist.append(rank)
        topic_dist = topic_dist/np.sum(topic_dist)

        out.append([word1, word2])
        print(topic_dist)

    for line in out:
        print(line)

prompt(4)

