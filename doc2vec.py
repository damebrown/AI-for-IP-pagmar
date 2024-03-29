from sklearn import utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import collections
import pandas as pd
import gensim

tqdm.pandas(desc = "progress-bar")

PROCESSED_DATA_PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\data\\'
string_cols = [
    'Text',
    'text broken to words']
numeric_cols = ['titles', 'hour', 'day', 'month', 'year', 'class', '# of words', '# of rows', '# of houses',
                'longest row', 'shortest row']
cols = numeric_cols + string_cols
GREEN = 'green.csv'


def simple_preprocess(doc, deacc = False, min_len = 2, max_len = 15):
    tokens = [
        # token for token in gensim.utils.tokenize(doc, lower = True, deacc = deacc, encoding = 'utf8', errors = 'ignore')
        gensim.utils.to_utf8(token) for token in
        gensim.utils.tokenize(doc, lower = True, deacc = deacc, encoding = 'utf8', errors = 'ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def read_corpus(data, label, tokens_only = False):
    docs = data.iloc[:, 9]
    np_docs = docs.to_numpy()
    arr = []
    for i, line in enumerate(np_docs):
        if i:
            spp = simple_preprocess(line)
            arr.append(gensim.models.doc2vec.TaggedDocument(spp, [label + str(i)]))
    return arr


def extract_vec(df, model):
    df['doc2vec'] = ''
    count_row = df.shape[0]
    j = df.columns.get_loc('doc2vec')
    w = df.columns.get_loc('text broken to words')
    words = df[df.columns[w]]
    h = words.to_numpy()
    for i in range(count_row):
        word = words[i].strip('[]').replace(',', '').split()
        df.iat[i, w] = word
        df.iat[i, j] = model.infer_vector(word)
    df.to_csv(PROCESSED_DATA_PATH + 'green.csv', encoding = 'utf-8-sig')
    return df


all_df = pd.read_csv(PROCESSED_DATA_PATH + 'all_data.csv', names = cols)
green_df = pd.read_csv(PROCESSED_DATA_PATH + 'green.csv', names = cols)
all_np = all_df.to_numpy()
green_np = green_df.to_numpy()
green_df = green_df.iloc[1:]
model = gensim.models.Doc2Vec.load('doc2vec_model')
green_df['avg line len'] = (green_df['# of words'].astype(int) / green_df['# of rows'].astype(int)).astype(float)
print(green_df.columns)
green_df = extract_vec(green_df, model)
print(1)
# extract_vec(green_df, green_df[green_df.columns[12]], model)

# train, test = train_test_split(df, test_size = 0.2, random_state = 42)
# train_corpus = list(read_corpus(train, 'train'))
# test_corpus = list(read_corpus(test, 'test', tokens_only = True))
# all_data = test_corpus + train_corpus
# model.build_vocab([x for x in tqdm(all_data)])
# model.train(all_data, total_examples = len(all_data), epochs = model.epochs)
# for epoch in range(30):
#     model.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples = len(all_data), epochs = 1)
#     model.alpha -= 0.002
#     model.min_alpha = model.alpha
# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn = len(model.docvecs))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)
#
#     second_ranks.append(sims[1])
#     print(collections.Counter(ranks))
#
#     print('Document ({}): {}\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
#     print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
#     for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
#         print(u'%s %s: %s\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

#
# vocabulary = list(model.wv.vocab)
# _X = model[vocabulary]
#
#
# def plot_tsne(X, vocab):
#     tsne = TSNE(n_components = 2)
#     X_tsne = tsne.fit_transform(X)
#     print("processing data")
#     df = pd.DataFrame(X_tsne, index = vocab, columns = ['x', 'y'])
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#
#     print("plots scatter space for TSNe")
#     ax.scatter(df['x'], df['y'])
#     for word, pos in df.iterrows():
#         ax.annotate(word[::-1], pos)
#     print("showing")
#     plt.show()
#
#
# def plot_pca(X, vocab):
#     pca = PCA(n_components = 2)
#     X_pca = pca.fit_transform(X)
#     print("processing data")
#     df = pd.DataFrame(X_pca, index = vocab, columns = ['x', 'y'])
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#
#     print("plots scatter space for PCA")
#     ax.scatter(df['x'], df['y'])
#     for word, pos in df.iterrows():
#         ax.annotate(word[::-1], pos)
#     print("showing")
#     plt.show()
