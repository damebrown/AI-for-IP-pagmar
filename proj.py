from pprint import pprint  # pretty-printer
from sklearn.decomposition import PCA
from collections import defaultdict
import gensim
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

from_keep = 'C:\\Users\\user\\PycharmProjects\\AIFIP_final\\songs_from_keep.txt'


def build_model(sentences):
    print("starting to build model")
    model = gensim.models.Word2Vec(sentences)
    print("model built. saving...")
    model.save('poems_keep_model.bin')
    print("done saving")
    return model


def get_sentences(path):
    arr = []
    print("getting sentences")
    # with open(path, 'r') as f:
    # df = pd.read_csv('output_list.txt', sep='\n')
    file = open(path, 'r', encoding = "utf8")
    lines = file.readlines()
    for line in lines:
        if line != '\n':
            line = line.strip('.-?!,\n')
            arr.append(line)
    print("done getting sentences")
    return arr


def count_enters(poem):
    poem = poem.split('\n')
    number_of_words = 0
    number_of_rows = len(poem)
    longest_row_length = 0
    shortest_row_length = 400
    for i in range(len(poem)):
        row = poem[i].split(' ')
        len_row = len(row)
        number_of_words += len_row
        if longest_row_length < len_row:
            longest_row_length = len_row
        if shortest_row_length > len_row:
            shortest_row_length = len_row
        poem[i] = row


# stoplist = set('for a of the and to in'.split())
# sentences = [[word for word in sentence.lower().split() if word not in stoplist] for sentence in sentences]
sentences = get_sentences(from_keep)

sentences = [[word for word in sentence.lower().split()] for sentence in sentences]
# remove words that appear only once
frequency = defaultdict(int)
for text in sentences:
    for token in text:
        frequency[token] += 1

# model = build_model(sentences)
# model = gensim.models.Word2Vec.load('poems_keep_model.bin')
model = gensim.models.Word2Vec(sentences)
print("taking vocab")
vocab = list(model.wv.vocab)
X = model[vocab]
tsne = TSNE(n_components = 2)
X_tsne = tsne.fit_transform(X)
print("processing data")
df = pd.DataFrame(X_tsne, index = vocab, columns = ['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

print("plots scatter space for TSNe")
ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word[::-1], pos)
print("showing")
plt.show()

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)
print("processing data")
df = pd.DataFrame(X_pca, index = vocab, columns = ['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

print("plots scatter space for PCA")
ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word[::-1], pos)
print("showing")
plt.show()

# X = model[model.wv.vocab]
# pca = PCA(n_components = 2)
# result = pca.fit_transform(X)

w2c = dict()
for i, word in enumerate(vocab):
    # plt.annotate(word, xy = (result[i, 0], result[i, 1]))
    w2c[word] = model.wv.vocab[word].count
w2cSorted = dict(sorted(w2c.items(), key = lambda x: x[1], reverse = True))
# plt.scatter(w2cSorted.keys(), w2cSorted.values())
print(w2cSorted)
w2cSortedList = reversed(list(w2cSorted.keys()))

print(w2cSortedList)
plt.show()
for word in w2cSortedList:
    print("similarity with ", word, " is ", model.similarity('אור', word))
