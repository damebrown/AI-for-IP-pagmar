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
NOTEBOOK_PATH = 'C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\'
YESHIVA_1 ='yeshiva_1.csv'
YESHIVA_2 ='yeshiva_2.csv'


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
    # pd_poems = pd.read_csv(path)
    poems = pd_poems.to_numpy()
    poems[:, 0] = pd_poems.iloc[:, 0]
    for poem in pd_poems.iterrows():
        print(poem[1][0])
        poem_texts.append(poem[1][0])
        month = poem[1][1]
        year = poem[1][2]
        hours.append(-1)
        days.append(-1)
        months.append(month)
        years.append(year)
        titles.append(poem[1][3])
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
    df.to_csv(NOTEBOOK_PATH + YESHIVA_2, encoding = 'utf-8-sig')
    return df


string_cols = [
    'Text',
    'text broken to words']
numeric_cols = ['titles', 'hour', 'day', 'month', 'year', 'class', '# of words', '# of rows', '# of houses',
                'longest row', 'shortest row']
cols = numeric_cols + string_cols
filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

# TODO- doc 2 vec
# df = parse_keep()
# df = pd.read_csv(NUMERIC_PATH, names = cols)
# np_df = df.to_numpy()
# np_df = np_df[1:, :-2].astype(int)
# bins = max(np_df[:, 5])
# plt.hist(np_df[:, 5], bins)
# plt.grid(axis = 'x')
# plt.xticks(np.arange(12, 78, step = 12))
# plt.show()

path_18_19 = "C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\notebooks\\poems_18_19.csv"
path_yeshiva_1 = "C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\notebooks\\yeshive_poems_1.csv"
path_yeshiva_2 = "C:\\Users\\user\\PycharmProjects\\AIFIP_pagmar\\notebooks\\yeshive_2_and_more.csv"
df = parse_notebooks(path_yeshiva_2)
npdf = df.to_numpy()
print(1)
