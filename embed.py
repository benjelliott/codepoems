import nltk
import string
import numpy as np
from scipy import sparse
import tqdm
f = open('book.txt','r')
text = f.read()
f.close()
token = nltk.word_tokenize(text)
token = list(filter(lambda token: token not in string.punctuation, token))
window = 3
pairs = []
length = len(token)
for i, word in enumerate(token):
    for j in range(-window, window + 1):
        if i < 3:
            if i + j > 0:
                pairs.append([token[i], token[i+j]])
        elif i > length - window - 1:
            if i + j < length - 1 and j != 0:
                pairs.append([token[i], token[i+j]])
        else:
            if j != 0:
                pairs.append([token[i], token[i+j]])
unique = list(set(token))
unique.sort()
unidict = {}
for i, word in enumerate(unique):
    unidict.update({word: i})
n = len(pairs)
m = len(unidict)
words = list(unidict.keys())
X = []
Y = []
for i, pair in tqdm.tqdm(enumerate(pairs)):
    focus_index = unidict.get(pair[0])
    context_index = unidict.get(pair[1])
    X_row = np.zeros(m)
    Y_row = np.zeros(m)
    X_row[focus_index] = 1
    Y_row[context_index] = 1
    X.append(X_row)
    Y.append(Y_row)
X = np.asarray(X)
Y = np.asarray(Y)
# Deep learning: 
from keras.models import Input, Model
from keras.layers import Dense

# Defining the size of the embedding
embed_size = 2

# Defining the neural network
inp = Input(shape=(X.shape[1],))
x = Dense(units=embed_size, activation='linear')(inp)
x = Dense(units=Y.shape[1], activation='softmax')(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Optimizing the network weights
model.fit(
    x=X, 
    y=Y, 
    batch_size=256,
    epochs=1000
    )

# The input layer 
weights = model.get_weights()[0]

# Creating a dictionary to store the embeddings in. The key is a unique word and 
# the value is the numeric vector
embedding_dict = {}
for word in words: 
    embedding_dict.update({
        word: weights[unique_word_dict.get(word)]
        })
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for word in list(unique_word_dict.keys()):
  coord = embedding_dict.get(word)
  plt.scatter(coord[0], coord[1])
  plt.annotate(word, (coord[0], coord[1]))