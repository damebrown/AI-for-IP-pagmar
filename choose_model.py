import gensim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

PROCESSED_DATA_PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\data\\'

string_cols = [
    'Text',
    'text broken to words', 'avg line len', 'poem2vec']
numeric_cols = ['titles', 'hour', 'day', 'month', 'year', 'class', '# of words', '# of rows', '# of houses',
                'longest row', 'shortest row']
cols = numeric_cols + string_cols


def parse_data(all_df):
    npdf = all_df.to_numpy()
    df = all_df.drop(all_df.index[0])
    vectors = df[df.columns[12]]
    vectors = vectors.to_numpy()
    df = df.drop(df.columns[12], axis = 1)
    df = df.astype(float)
    npdf = df.to_numpy()
    length = len(df.columns)
    for i in range(50):
        label = 'vec' + str(i)
        df[label] = np.zeros_like(vectors)
    arr = np.ndarray(shape = (len(vectors), 50))
    for j, vec in enumerate(vectors):
        l_vec = vec.replace(',', '').strip('[]').split()
        int_vec = [float(i) for i in l_vec]
        arr[j] = int_vec
    npdf = df.to_numpy()
    for i in range(50):
        col = length + i
        df[df.columns[col]] = arr[:, i]
        npdf = df.to_numpy()
    df = normalize(df)
    class_tag = df[df.columns[5]].astype(int)
    year_tag = df[df.columns[4]].astype(int)
    month_tag = df[df.columns[3]].astype(float)
    npdf = df.to_numpy()
    np_year = year_tag.to_numpy()
    np_month = month_tag.to_numpy()
    df = df.drop(df.columns[5], axis = 1)
    df = df.drop(df.columns[4], axis = 1)
    df = df.drop(df.columns[3], axis = 1)
    return class_tag, year_tag, month_tag, df


def normalize(df):
    old_np = df.to_numpy()
    # title
    old_np[:, 0] -= 0.5
    # hour
    old_np[:, 1] = (old_np[:, 1] - 11) / 12.0
    # day
    old_np[:, 2] = (old_np[:, 2] - 15) / 16.0
    # # month
    old_np[:, 3] = (old_np[:, 3] - 6.5).astype(float)
    # old_np[:, 3] = (old_np[:, 3] - 6.5) / 5.5
    # # year
    old_np[:, 4] = (old_np[:, 4] - 16)
    # old_np[:, 4] = (old_np[:, 4] - 16) / 3.0
    # class
    old_np[:, 5] = (old_np[:, 5] - 43)
    # # of words
    old_np[:, 6] = (old_np[:, 6] - 381) / 376.0
    # # of rows
    old_np[:, 7] = (old_np[:, 7] - 28) / 27.0
    # # of houses
    old_np[:, 8] = (old_np[:, 8] - 6) / 5.0
    # longest
    old_np[:, 9] = (old_np[:, 9] - 381) / 376.0
    # # of rows
    old_np[:, 10] = (old_np[:, 10] - 28) / 27.0
    # # of houses
    old_np[:, 11] = (old_np[:, 11] - 6) / 5.0
    # vecs
    for i in range(12, 62):
        minima = min(old_np[:, i])
        maxima = max(old_np[:, i])
        old_np[:, i] = (old_np[:, i] - (maxima + minima) / 2.0) / ((maxima - minima) / 2.0)
    return pd.DataFrame(old_np)


def get_united_data():
    all_df = pd.read_csv(PROCESSED_DATA_PATH + 'all_data.csv', names = cols)
    green_df = pd.read_csv(PROCESSED_DATA_PATH + 'green.csv', names = cols)
    df = all_df.append(green_df)
    return df


# df = get_united_data()
# df.to_csv(PROCESSED_DATA_PATH + 'united_data.csv')
# npdf = df.to_numpy()
# drop_cols = [11, 12]
# df = df.drop(df.columns[drop_cols], axis = 1)
# before_npdf = df.to_numpy()
# text, words = df['Text'], df['text broken to words']
# text.to_csv(PROCESSED_DATA_PATH + 'text.csv')
# words.to_csv(PROCESSED_DATA_PATH + 'broken_to_words.csv')

# class_y, year_y, month_y, df = parse_data(df)
#
# after_npdf = df.to_numpy()
# np_year = year_y.to_numpy()
# np_month = month_y.to_numpy()
#
# df.to_csv(PROCESSED_DATA_PATH + 'unitd_numeric_data.csv')
# month_y.to_csv(PROCESSED_DATA_PATH + 'month_y.csv')
# year_y.to_csv(PROCESSED_DATA_PATH + 'year_y.csv')
# class_y.to_csv(PROCESSED_DATA_PATH + 'class_y.csv')
#
# df = pd.read_csv(PROCESSED_DATA_PATH + 'unitd_numeric_data.csv')
# month_y = pd.read_csv(PROCESSED_DATA_PATH + 'month_y.csv')
# year_y = pd.read_csv(PROCESSED_DATA_PATH + 'year_y.csv')
# class_y = pd.read_csv(PROCESSED_DATA_PATH + 'class_y.csv')
# random_state = 170


# class_y = np.floor(class_y / 3)
# month_y = np.floor(month_y / 3)
# npy = class_y.to_numpy()
# npdf = df.to_numpy()
# labels = [year_y, month_y]
# strings = ["year", "month"]

# for i, label in enumerate(labels):
#     print("========================" + strings[i] + "=======================")
#     test_size = 0.02
#     while test_size < 0.07:
#         print("~~~~~~~~~~~~TEST SIZE IS " + str(test_size))
#         X_train, X_test, y_train, y_test = train_test_split(df, label, test_size = test_size, random_state = None)
#         npX_train, npX_test, npy_train, npy_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
#
#         models = {
#             "forest": RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 0),
#             "tree": DecisionTreeClassifier(max_depth = 3, random_state = 0),
#             "svc": SVC(gamma = 'auto'),
#             "bagging": BaggingClassifier(n_estimators = 100, random_state = 0),
#             "extraTree": ExtraTreesClassifier(n_estimators = 100, random_state = 0),
#             "Naive Bayes": GaussianNB(),
#             "Logistic Reg": LogisticRegression(),
#             "LDA": LinearDiscriminantAnalysis(),
#         }
#         scores = dict()
#         maximum, maximum_boost, max_name, max_boost_name = 0, 0, " ", " "
#         for name, model in models.items():
#             success = []
#             for i in range(10, 101, 20):
#                 model.fit(X_train, y_train)
#                 score = model.score(X_test, y_test)
#                 success.append(score)
#             avg = sum(success) / len(success)
#             if avg > maximum:
#                 maximum = avg
#                 max_name = name
#         print("best success rate = " + str(maximum) + "%, by " + max_name)
#         for name, model in models.items():
#             success = []
#             if name != "LDA" and name != "Linear Reg":
#                 for i in range(10, 101, 20):
#                     booster = AdaBoostClassifier(base_estimator = model, n_estimators = i, learning_rate = 1,
#                                                  algorithm = 'SAMME')
#                     booster.fit(X_train, y_train)
#                     score = booster.score(X_test, y_test)
#                     success.append(score)
#                 avg = sum(success) / len(success)
#                 if avg > maximum_boost:
#                     maximum_boost = avg
#                     max_boost_name = name
#         print("\tboosting best success rate = " + str(maximum_boost) + "%, by " + max_boost_name)
#         test_size += 0.01

def simple_preprocess(doc, deacc = False, min_len = 2, max_len = 15):
    tokens = [
        token for token in gensim.utils.tokenize(doc, lower = True, deacc = deacc, encoding = 'utf8', errors = 'ignore')
        # gensim.utils.to_utf8(token) for token in gensim.utils.tokenize(doc, lower = True, deacc = deacc, encoding = 'utf8', errors = 'ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def read_corpus(data, label):
    # docs = data.iloc[:, 9]
    # np_docs = docs.to_numpy()
    np_docs = data.to_numpy()
    arr = []
    for i, line in enumerate(np_docs):
        # if i:
        spp = simple_preprocess(line[1])
        arr.append(gensim.models.doc2vec.TaggedDocument(spp, [label + str(i)]))
    return arr


def plot_pca(X, vocab, dim):
    pca = PCA(n_components = dim)
    X_pca = pca.fit_transform(X)
    print("processing data")
    df = pd.DataFrame(X_pca, index = vocab, columns = ['x', 'y'])
    # pca.fit(X)
    # result = pd.DataFrame(pca.transform(X), columns = ['PCA%i' % i for i in range(3)])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax = fig.add_subplot(111, projection = '3d')

    print("plots scatter space for PCA")
    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word[::-1], pos)
    print("showing")
    plt.show()


def plot_tsne(X, vocab, dim):
    tsne = TSNE(n_components = dim)
    X_tsne = tsne.fit_transform(X)
    print("processing data")
    df = pd.DataFrame(X_tsne, columns = ['x', 'y', 'z'])
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    ax = fig.add_subplot(111, projection = '3d')

    print("plots scatter space for TSNe")
    ax.scatter(df['x'], df['y'], df['z'])
    for word, pos in df.iterrows():
        ax.annotate(word[::-1], pos)
    print("showing")
    plt.show()


def init_data():
    df = pd.read_csv(PROCESSED_DATA_PATH + 'unitd_numeric_data.csv')
    month_y = pd.read_csv(PROCESSED_DATA_PATH + 'month_y.csv')
    year_y = pd.read_csv(PROCESSED_DATA_PATH + 'year_y.csv')
    # class_y = pd.read_csv(PROCESSED_DATA_PATH + 'class_y.csv')
    np_year = year_y.to_numpy()[:, 1]
    np_month = month_y.to_numpy()[:, 1]
    npdf = df.to_numpy()
    poem_index = 336
    int1, int2, int3, int4, int5 = np.random.randint(0, len(df), 5)
    print(str(int1) + ", " + str(int2) + ", " + str(int3) + ", " + str(int4) + ", " + str(int5) + ", " + str(
        poem_index))
    ints = [int1, int2, poem_index]
    df = df.drop(df.index[ints])
    year_y = year_y.drop(df.index[ints])
    month_y = month_y.drop(df.index[ints])
    poems = df.iloc[ints].to_numpy()
    return [year_y, month_y], df, poems


# model = gensim.models.Doc2Vec.load('doc2vec_model')
# vocab = model.wv.vocab
# # X = model[vocab]
# vec = model.docvecs
# dim = 2
# # plot_pca(X, vocab, dim)
# # plot_tsne(X, vocab, dim)
#
# from matplotlib.colors import ListedColormap
#
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# strings = ["month", "year"]
#
# h = .02  # step size in the mesh
# labels, numeric_df, poems = init_data()
# np_df = numeric_df.to_numpy()[:, 1:]
# np_year = labels[0].to_numpy()[:, 1:]
# np_month = labels[1].to_numpy()[:, 1:]
# for i, label in enumerate(labels):
#     print("========================" + strings[i] + "=======================")
#     for k_neighbors in range(1, 10):
#         for weights in ['uniform', 'distance']:
#             print("  \tKNN WITH " + str(k_neighbors) + " NEIGHBORS")
#             knn = KNeighborsClassifier(n_neighbors = k_neighbors, weights = weights)
#             knn.fit(numeric_df, label)
#             # print(knn.score(numeric_df, label))
#             pca = PCA(n_components = 2)
#             X_pca = pca.fit_transform(numeric_df)
#
#             print("processing data")
#             X = pd.DataFrame(X_pca, columns = ['x', 'y']).to_numpy()
#             # Y = pd.DataFrame(Y_pca, columns = ['x', 'y', 'z']).to_numpy()
#
#             x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#             y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#             # z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
#             xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                                  np.arange(y_min, y_max, h))
#             # Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
#             a = np.c_[xx.ravel(), yy.ravel()]
#             Z = knn.predict(a)
#
#             # Put the result into a color plot
#             Z = Z.reshape(xx.shape)
#             fig = plt.figure()
#             ax = fig.add_subplot(projection = '3d')
#             plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
#
#             # Plot also the training points
#             plt.scatter(X[:, 0], X[:, 1], X[:, 2], c = label, cmap = cmap_bold,
#                         edgecolor = 'k', s = 20)
#             plt.xlim(xx.min(), xx.max())
#             plt.ylim(yy.min(), yy.max())
#             plt.title("3-Class classification (k = %i, weights = '%s')" % (i, weights))
#
#     plt.show()


def when(poem):
    if poem == "ISLAND":
        print("POEM IS: " + poem)
    elif poem == "OVER":
        print("POEM IS: " + poem)
    elif poem == "STAV":
        print("POEM IS: " + poem)
    return 0
