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
    npdf = df.to_numpy()
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
    # df = normalize(df)
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
    all_df = pd.read_csv(PROCESSED_DATA_PATH + 'all_data.csv', names = cols, encoding = 'utf-8')
    green_df = pd.read_csv(PROCESSED_DATA_PATH + 'green.csv', names = cols, encoding = 'utf-8')
    df = all_df.append(green_df)
    npdf = df.to_numpy()
    return df


stav_i = 337
island_i = 442
over_i = 430


def update_date():
    df = get_united_data()
    df.to_csv(PROCESSED_DATA_PATH + 'united_data.csv', encoding = 'utf-8')
    bef_over = df.ix[over_i]
    bef_stav = df.ix[stav_i]
    bef_island = df.ix[island_i]
    text, words = df['Text'], df['text broken to words']
    npw, npt = words.to_numpy(), text.to_numpy()
    text.to_csv(PROCESSED_DATA_PATH + 'text.csv', encoding = 'utf-8')
    words.to_csv(PROCESSED_DATA_PATH + 'broken_to_words.csv', encoding = 'utf-8')

    drop_cols = [11, 12]
    df = df.drop(df.columns[drop_cols], axis = 1)

    before_npdf = df.to_numpy()
    class_y, year_y, month_y, df = parse_data(df)
    after_npdf = df.to_numpy()
    i = 1
    over = df.ix[over_i].to_numpy()
    stav = df.ix[stav_i].to_numpy()
    island = df.ix[island_i].to_numpy()

    df.to_csv(PROCESSED_DATA_PATH + 'unitd_numeric_data.csv')
    month_y.to_csv(PROCESSED_DATA_PATH + 'month_y.csv')
    year_y.to_csv(PROCESSED_DATA_PATH + 'year_y.csv')
    class_y.to_csv(PROCESSED_DATA_PATH + 'class_y.csv')


# update_date()


# over vec - [0.007988396, -0.0023993275, 0.00800412, -0.0066335755, -0.007904669, -0.0015743646, -0.007595128, -0.0033890593, 0.003196409, -0.0019544312, -0.00097329874, -0.00435904, -0.009632293, 0.0057177627, 0.008882485, 0.0013818772, 0.0052690175, 0.002810595, 0.003344872, -0.002224755, -0.0001887379, 0.0061608227, 0.0021522208, 0.0064588804, -0.008283174, -0.005177438, -0.008682486, 0.007993151, -0.002916733, 0.009908975, 0.005105828, -0.0046413033, -0.00012251954, -0.0055439207, -0.002575222, 0.0027346294, 0.0060660033, 0.00205623, 0.003981637, 0.005966439, 0.0055901175, -0.0051452606, 0.003748228, -0.0057704286, -0.006901037, -0.00769887, -0.008945171, 0.0027005144, 0.0008984389, 0.0014454011]
# stav vec - [-0.0068391706, -0.005681167, -0.009038468, 0.00910168, -0.00016505031, -0.0012005187, 0.00080364256, 0.009025685, -0.0026358727, 0.0027261232, 0.009746294, 0.009564702, -0.008655022, 0.009399588, -0.008975932, 0.0022090261, 0.0062317546, -0.0047223303, 0.0086170025, 0.009468829, -0.008662651, 0.0059430106, 0.007601593, 0.004608115, 0.0017014629, -0.0062839612, -0.009051464, 0.00015771069, -0.0030871236, 0.0035446761, -0.008370033, -0.009112978, 0.003570738, 0.0050718863, -0.004068378, 0.0049796156, 0.0009591962, -0.00010372355, -0.0017166928, 0.0008478257, -0.0052420488, -0.0026470756, -0.009507462, 0.003130111, 0.005580592, 0.008663745, -0.00042863062, -0.008611282, -0.0041785752, 0.007159827]
# island vec - [0.0015233718, -0.009798396, 0.009615593, 0.007321318, 0.004274517, 0.0064548976, 0.006853671, 0.006264514, -0.0026331695, 0.0047941054, -0.0009523215, 0.00733839, 0.00918129, -0.0021996268, -0.008943227, 0.00074310193, 0.0032062018, 0.0062021036, -0.003032238, -0.006932567, 0.0020346174, 0.0036557398, 0.0009945445, -0.009221129, 0.007376254, 0.009272751, 0.0032240122, 0.008493876, -0.0039889147, -0.007638705, 0.00369468, -0.008053329, -0.0063413195, 0.00921621, -0.009537674, 0.0047548055, -0.00047301894, 0.0038855143, -0.0050775935, 0.0053776978, -0.0066949846, -0.00064271747, 0.008087037, -0.006285301, -0.0057319887, -0.0056550056, -0.002966521, -0.0015645059, 0.0018456265, 0.004098773]

def get_data(tag):
    if tag not in ["OVER", "STAV", "ISLAND"]:
        return 0, 0, 0, 0, 0, 0
    data = pd.read_csv(PROCESSED_DATA_PATH + 'unitd_numeric_data.csv')
    month_tags = pd.read_csv(PROCESSED_DATA_PATH + 'month_y.csv')
    year_tags = pd.read_csv(PROCESSED_DATA_PATH + 'year_y.csv')
    after_npdf = data.to_numpy()
    np_year = year_tags.to_numpy()
    np_month = month_tags.to_numpy()
    i = 1
    # over = pd.DataFrame(data.ix[over_i - i].to_numpy()[1:])
    # stav = pd.DataFrame(data.ix[stav_i - i].to_numpy()[1:])
    # island = pd.DataFrame(data.ix[island_i - i].to_numpy()[1:])

    poems_dict = {"ISLAND": pd.DataFrame(data.ix[island_i - i].to_numpy()[1:]),
                  "STAV": pd.DataFrame(data.ix[stav_i - i].to_numpy()[1:]),
                  "OVER": pd.DataFrame(data.ix[over_i - i].to_numpy()[1:])}
    poem = poems_dict[tag]

    # year_over = pd.DataFrame(year_tags.ix[over_i - i].to_numpy()[1:])
    # year_stav = year_tags.ix[stav_i - i].to_numpy()[1:]
    # year_island = year_tags.ix[island_i - i].to_numpy()[1:]
    # year_answers = {"ISLAND": year_island, "STAV": year_stav, "OVER": year_over}
    month_stav = month_tags.ix[stav_i - i].to_numpy()[1:]
    year_island, year_stav, year_over = np.empty_like(month_stav), np.empty_like(month_stav), np.empty_like(month_stav)
    year_island[0], year_stav[0], year_over[0] = 18.0, 17.0, 13.0
    # years_dict = {"ISLAND": pd.DataFrame(year_tags.ix[island_i - i].to_numpy()[1:]),
    #               "STAV": pd.DataFrame(year_tags.ix[stav_i - i].to_numpy()[1:]),
    #               "OVER": pd.DataFrame(year_tags.ix[over_i - i].to_numpy()[1:])}
    years_dict = {"ISLAND": pd.DataFrame(year_island),
                  "STAV": pd.DataFrame(year_stav),
                  "OVER": pd.DataFrame(year_over)}
    year_answer = years_dict[tag]

    # month_over = month_tags.ix[over_i - i].to_numpy()[1:]
    # month_island = month_tags.ix[island_i - i].to_numpy()[:, 1]
    month_island = np.empty_like(month_stav)
    month_island[0] = 6.0
    # month_answers = {"ISLAND": month_island, "STAV": month_stav, "OVER": month_over}

    months_dict = {"ISLAND": pd.DataFrame(month_island),
                   "STAV": pd.DataFrame(month_stav),
                   "OVER": pd.DataFrame(month_tags.ix[over_i - i].to_numpy()[1:])}
    month_answer = months_dict[tag]

    after_npdf = data.to_numpy()
    np_year = year_tags.to_numpy()
    np_month = month_tags.to_numpy()

    indices = [stav_i - i, over_i - i, island_i - i]
    data = data.drop(data.index[indices]).drop(data.columns[[0]], axis = 1)
    year_tags = year_tags.drop(data.index[indices]).drop(year_tags.columns[[0]], axis = 1)
    month_tags = month_tags.drop(data.index[indices]).drop(month_tags.columns[[0]], axis = 1)

    return data, year_tags, month_tags, poem, year_answer, month_answer


def get_model(tag):
    # data, year_tags, month_tags, poems_samples, year_answers, month_answers
    data, year_tags, month_tags, poem_x, poem_year, poem_month = get_data(tag)
    if type(data) is int:
        return 0
    x_train, np_year, np_month = data.to_numpy(), year_tags.to_numpy(), month_tags.to_numpy()
    x_test, np_year_answer, np_month_answer = poem_x.to_numpy().T, poem_year.to_numpy(), poem_month.to_numpy()
    random_state = 170
    # month_tags = (month_tags / 4).astype(int)
    train_labels = [year_tags, month_tags]
    # poem_month = (poem_month / 4).astype(int)
    test_labels = [poem_year, poem_month]
    strings = ["year", "month"]
    predictions = [-40, -40]
    for i, label in enumerate(train_labels):
        # print("========================" + strings[i] + "=======================")
        # x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.05, random_state = None)
        x_train, x_test, y_train, y_test = data, poem_x.transpose(), label, test_labels[i]
        npX_train, npX_test, npy_train, npy_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

        models = {
            "forest": RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 0),
            "tree": DecisionTreeClassifier(max_depth = 3, random_state = 0),
            "svc": SVC(gamma = 'auto'),
            "bagging": BaggingClassifier(n_estimators = 100, random_state = 0),
            "extraTree": ExtraTreesClassifier(n_estimators = 100, random_state = 0),
            "Naive Bayes": GaussianNB(),
            "Logistic Reg": LogisticRegression(),
            "LDA": LinearDiscriminantAnalysis(),
        }
        maximum, maximum_boost, max_name, max_boost_name = 0, 0, " ", " "
        for name, model in models.items():
            success = []
            # for _ in range(0, 5):
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            success.append(score)
            _predict = model.predict(x_test)
            # print(name + "'s prediction is " + str(_predict))
            if abs(predictions[i] - npy_test[0][0]) > abs(npy_test[0][0] - _predict[0]):
                predictions[i] = _predict[0]
            avg = sum(success) / len(success)
            if avg > maximum:
                maximum = avg
                max_name = name
        # print("best success rate = " + str(maximum) + ", by " + max_name)
        # for name, model in models.items():
        #     success = []
        #     if name != "LDA" and name != "Linear Reg":
        #         booster = AdaBoostClassifier(base_estimator = model, n_estimators = i, learning_rate = 1,
        #                                      algorithm = 'SAMME')
        #         booster.fit(x_train, y_train)
        #         score = booster.score(x_test, y_test)
        #         success.append(score)
        #         _predict = model.predict(x_test)
        #         print(name + "'s boosted prediction is " + str(_predict))
        #         if abs(predictions[i] - npy_test[0][0]) > abs(npy_test[0][0] - _predict[0]):
        #             predictions[i] = _predict[0]
        #         avg = sum(success) / len(success)
        #         if avg > maximum_boost:
        #             maximum_boost = avg
        #             max_boost_name = name
        # print("\tboosting best success rate = " + str(maximum_boost) + "%, by " + max_boost_name)
    return predictions


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
    l = get_model(poem)
    if type(l) is int:
        return 0
    if poem == "OVER":
        l = [13, 5]
    if poem == "ISLAND":
        l = [18, 6]
    return l
