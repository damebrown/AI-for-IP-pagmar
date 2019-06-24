import functools
import json
import operator
import matplotlib.pyplot as plt
import gkeepapi
import pandas as pd
from pandas import plotting
import csv
import numpy as np
import seaborn as sns

NUMERIC_PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\numeric_data.csv'
PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\data.csv'
PROCESSED_DATA_PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\data\\'
NOTEBOOKS_PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\notebooks\\'
YESHIVA_1 = 'yeshiva_1.csv'
YESHIVA_2 = 'yeshiva_2.csv'
YESHIVA_3 = 'yeshiva_3.csv'
GREEN = 'green.csv'


def update_keep(email, password):
    keep = gkeepapi.Keep()
    keep.login(username = email, password = password)
    keep.sync()
    return keep


def save_keep(keep):
    # save file
    keep.sync()
    state = keep.dump()
    fh = open('state', 'w')
    json.dump(state, fh)


def resume_last_keep():
    keep = gkeepapi.Keep()
    fh = open('state', 'r')
    state = json.load(fh)
    keep.restore(state)
    return keep


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


nikkud = ['-', '?', '!', '.', '', ' ', '\n']


def count_enters(poem):
    poem = poem.split('\n')
    n_words = 0
    n_rows = 0
    n_houses = 1
    longest_row_length = 0
    shortest_row_length = 400
    broken_poem = []
    for i in range(len(poem)):
        if poem[i] == '':
            n_houses += 1
        else:
            n_rows += 1
            row = poem[i].strip('.-?!,\n')
            row = row.split(' ')
            row = list(filter(None, row))
            len_row = len(row)
            n_words += len_row
            if longest_row_length < len_row:
                longest_row_length = len_row
            if shortest_row_length > len_row:
                shortest_row_length = len_row
            broken_poem.append(row)
    broken_poem = functools.reduce(operator.iconcat, broken_poem, [])
    indices_to_remove = []
    for i in range(len(broken_poem)):
        if broken_poem[i] in nikkud:
            indices_to_remove.append(i)
        broken_poem[i] = broken_poem[i].strip('.-?!,\n')
        if broken_poem[i].startswith('-') or broken_poem[i].endswith('-'):
            broken_poem[i] = broken_poem[i].replace('-', '')
        broken_poem[i] = broken_poem[i].replace('?', '')
        broken_poem[i] = broken_poem[i].replace('!', '')
        broken_poem[i] = broken_poem[i].replace('.', '')
        broken_poem[i] = broken_poem[i].replace(',', '')
        broken_poem[i] = broken_poem[i].replace(';', '')
        broken_poem[i] = broken_poem[i].replace(':', '')
        broken_poem[i] = broken_poem[i].replace(')', '')
        broken_poem[i] = broken_poem[i].replace('(', '')
        broken_poem[i] = broken_poem[i].replace('[', '')
        broken_poem[i] = broken_poem[i].replace(']', '')
    for i in reversed(indices_to_remove):
        broken_poem.remove(broken_poem[i])
    return n_rows, n_words, n_houses, longest_row_length, shortest_row_length, broken_poem


def parse_keep():
    keep = resume_last_keep()
    # keep = update_keep('danielbrown13@gmail.com', 'hevelhavalimfor5777')
    # save_keep(keep)
    notes = keep.all()
    counter = 0
    hours, days, months, years = [], [], [], []
    num_of_words, num_of_rows, num_of_houses, longest_row, shortest_row, classes = [], [], [], [], [], []
    titles, broken_2_words, poem_texts = [], [], []
    for note in notes:
        if note.labels._labels != {}:
            if note.labels._labels[u'tag.etny4hklimae.1560181267354']:
                counter += 1
                time = note.timestamps.created
                month = time.month
                year = time.year
                # n_rows, n_words, n_houses, longest_row_length, shortest_row_length, broken_poem
                poem_texts.append(note.text)
                a, b, c, d, e, f = count_enters(note.text)
                num_of_rows.append(a), num_of_words.append(b), num_of_houses.append(c)
                longest_row.append(d), shortest_row.append(e), broken_2_words.append(f)
                hours.append(time.hour)
                days.append(time.day)
                months.append(month)
                years.append(year - 2000)
                title = int(bool(note.title))
                titles.append(title)
                classes.append((year - 2013) * 12 + month)
                # print(time, hour, month, year, title)
    data = {
        'titles': titles,
        'hour': hours,
        'day': days,
        'month': months,
        'year': years,
        'class': classes,
        '# of words': num_of_words,
        '# of rows': num_of_rows,
        '# of houses': num_of_houses,
        'longest row': longest_row,
        'shortest row': shortest_row,
        'Text': poem_texts,
        'text broken to words': broken_2_words
    }
    df = pd.DataFrame(data, columns = cols)
    df.to_csv(PATH, encoding = 'utf-8-sig')
    df = df.drop(['Text'], axis = 1)
    df = df.drop(['text broken to words'], axis = 1)
    df.to_csv(NUMERIC_PATH, encoding = 'utf-8-sig')
    return df


def parse_notebooks(path):
    hours, days, months, years = [], [], [], []
    num_of_words, num_of_rows, num_of_houses, longest_row, shortest_row, classes = [], [], [], [], [], []
    titles, broken_2_words, poem_texts = [], [], []
    pd_poems = pd.read_csv(path, encoding = 'utf8')
    np_poems = pd_poems.to_numpy()
    for poem in pd_poems.iterrows():
        print("\n")
        print(poem[1][0])
        poem_texts.append(poem[1][0])
        month = poem[1][1]
        year = poem[1][2]
        hours.append(-1)
        days.append(-1)
        months.append(month)
        years.append(year)
        titles.append(poem[1][3])
        print(len(titles))
        classes.append((year - 13) * 12 + month)
        # n_rows, n_words, n_houses, longest_row_length, shortest_row_length, broken_poem
        a, b, c, d, e, f = count_enters(poem[1][0])
        num_of_rows.append(a), num_of_words.append(b), num_of_houses.append(c)
        longest_row.append(d), shortest_row.append(e), broken_2_words.append(f)
    data = {
        'titles': titles,
        'hour': hours,
        'day': days,
        'month': months,
        'year': years,
        'class': classes,
        '# of words': num_of_words,
        '# of rows': num_of_rows,
        '# of houses': num_of_houses,
        'longest row': longest_row,
        'shortest row': shortest_row,
        'Text': poem_texts,
        'text broken to words': broken_2_words
    }
    df = pd.DataFrame(data, columns = cols)
    df.to_csv(PROCESSED_DATA_PATH + GREEN, encoding = 'utf-8-sig')
    np_poems = df.to_numpy()
    return df


def unite_textual_data():
    arr = ['keep_data.csv', 'poems_18_19.csv', 'yeshiva_1.csv', 'yeshiva_2.csv', 'yeshiva_3.csv', GREEN]
    frames = []
    for file in arr:
        df = pd.read_csv(PROCESSED_DATA_PATH + file)
        frames.append(df)
    data = pd.concat(frames)
    data.to_csv(PROCESSED_DATA_PATH + 'all_data.csv', encoding = 'utf-8-sig')
    return data


string_cols = [
    'Text',
    'text broken to words']
numeric_cols = ['titles', 'hour', 'day', 'month', 'year', 'class', '# of words', '# of rows', '# of houses',
                'longest row', 'shortest row']
cols = numeric_cols + string_cols
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
df = parse_notebooks(NOTEBOOKS_PATH + GREEN)
npdf = df.to_numpy()
print(1)
# df = pd.read_csv(NUMERIC_PATH, names = cols)
# np_df = df.to_numpy()
# np_df = np_df[1:, :-2].astype(int)
# bins = max(np_df[:, 5])
# plt.hist(np_df[:, 5], bins)
# plt.grid(axis = 'x')
# plt.xticks(np.arange(12, 78, step = 12))
# plt.show()

# TODO extract Doc2Vec vector for each poems in all data sets (all notebooks and textual data from keep
# df = pd.read_csv(PROCESSED_DATA_PATH + 'all_data.csv', encoding = 'utf-8-sig')
# numeric_data = df.drop(df.columns[0], axis=1)
# # numeric_data = numeric_data.drop(['day'], axis=1)
# numeric_data['row avg length'] = numeric_data['# of words'] / numeric_data['# of rows']
# pd.plotting.scatter_matrix(numeric_data, alpha=0.2, figsize=(10, 10))
# plt.show()
# plt.matshow(numeric_data.corr())
# plt.show()
#
# cnt_pro = df['class'].value_counts()
# sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Class', fontsize=12)
# plt.xticks(rotation=90)
# plt.show()
# TODO remove all textual data
# TODO unify all data sets
