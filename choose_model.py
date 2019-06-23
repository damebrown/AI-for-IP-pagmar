import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals.six.moves import zip

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

PROCESSED_DATA_PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\data\\'

string_cols = [
    'Text',
    'text broken to words', 'avg line len', 'poem2vec']
numeric_cols = ['titles', 'hour', 'day', 'month', 'year', 'class', '# of words', '# of rows', '# of houses',
                'longest row', 'shortest row']
cols = numeric_cols + string_cols


def parse_data():
    orig_df = pd.read_csv(PROCESSED_DATA_PATH + 'all_data.csv', names = cols)
    df = orig_df.drop(orig_df.index[np.arange(450, 475)])
    npdf = df.to_numpy()
    df = df.drop(df.index[0])
    vectors, text, words = df['poem2vec'].to_numpy(), df['Text'], df['text broken to words']
    class_y = df[df.columns[5]]
    year_y = df[df.columns[4]]
    month_y = df[df.columns[3]]
    df = df.drop(df.columns[14], axis = 1)
    df = df.drop(df.columns[12], axis = 1)
    df = df.drop(df.columns[11], axis = 1)
    df = df.drop(df.columns[5], axis = 1)
    df = df.drop(df.columns[4], axis = 1)
    df = df.drop(df.columns[3], axis = 1)
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
    # df.to_csv(PROCESSED_DATA_PATH + 'all_data_numeric.csv')
    return class_y, year_y, month_y, df


class_y, year_y, month_y, df = parse_data()
npy = class_y.to_numpy()
npdf = df.to_numpy()
labels = [class_y, year_y, month_y]
strings = ["class", "year", "month"]
for i, label in enumerate(labels):
    print("==============" + strings[i] + "==============")
    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size = 0.2, random_state = None)
    npX_train, npX_test, npy_train, npy_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    models = {
        "forest": RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 0),
        "tree": DecisionTreeClassifier(max_depth = 3, random_state = 0),
        "svc": SVC(gamma = 'auto'),
        "bagging": BaggingClassifier(n_estimators = 100, random_state = 0),
        "extraTree": ExtraTreesClassifier(n_estimators = 100, random_state = 0),
        "Naive Bayes": GaussianNB(),
    }
    scores = dict()

    for name, model in models.items():
        scores[name] = []
        test_errors = []
        booster = AdaBoostClassifier(base_estimator = model, n_estimators = i, learning_rate = 1,
                                     algorithm = 'SAMME')
        booster.fit(X_train, y_train)
        for discrete_train_predict in zip(booster.staged_predict(X_test)):
            test_errors.append(1. - accuracy_score(discrete_train_predict, y_test))
        print(test_errors)
        # for i in tqdm(range(10, 101, 10), desc = name):
        #     scores[name].append(booster.score(X_test, y_test))
        #     prediction = booster.predict(X_test)
        #     mnp = y_test.to_numpy()
        #     counter = 0
        #     # print("\n")
        #     for j in range(len(prediction)):
        #         if mnp[j] != prediction[j]:
        #             counter += 1
        #             # print(text.at[j])
        #             # test_year = str(int(mnp[j]) / 12 + 2013)
        #             # test_month = str(int(mnp[j]) % 12 + 1)
        #             # pred_year = str(int(prediction[j]) / 12 + 2013)
        #             # pred_month = str(int(prediction[j]) % 12 + 1)
        #             # print(test_year + ", " + test_month + "--->" + pred_year + ", " + pred_month)
        #             # print('\n')
        #     print("======= error rate:" + str(float(counter) / float(len(mnp))) + "%")
        # plt.figure()
        # plt.title(name)
        # plt.plot(range(10, 101, 10), scores[name])
        # plt.show()

# scores = dict()
# model = BaggingClassifier(n_estimators = 100, random_state = 0)
# name = "bagging"
# scores[name] = []
# booster = 0
# for i in tqdm(range(10, 101, 10), desc = name):
#     booster = AdaBoostClassifier(base_estimator = model, n_estimators = i, learning_rate = 1,
#                                  algorithm = 'SAMME')
#     booster.fit(X_train, y_train)
#     booster_score = booster.score(X_test, y_test)
#     scores[name].append(booster_score)
#     prediction = booster.predict(X_test)
#     mnp = y_test.to_numpy()
#     counter = 0
#     print("\n")
#     for j in range(len(prediction)):
#         if mnp[j] != prediction[j]:
#             counter += 1
#             # print(text.at[j])
#             test_year = str(int(mnp[j]) / 12 + 2013)
#             test_month = str(int(mnp[j]) % 12 + 1)
#             pred_year = str(int(prediction[j]) / 12 + 2013)
#             pred_month = str(int(prediction[j]) % 12 + 1)
#             print(test_year + ", " + test_month + "--->" + pred_year + ", " + pred_month)
#             # print('\n')
#     print("======= number of errors:" + str(counter))
# print('\n')
# print(name + "'s scores are ", scores[name])
