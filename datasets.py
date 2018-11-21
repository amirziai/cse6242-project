import csv
from typing import List

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

DATASET_PATH = 'resources/datasets/'
MAX_DOCS = 2000


def get_bbc() -> List[str]:
    docs = [line[1] for i, line in enumerate(csv.reader(open(f"{DATASET_PATH}bbc-text.csv"), delimiter=',')) if i > 0]
    return docs[:MAX_DOCS]


# articles1.csv from https://www.kaggle.com/snapcrack/all-the-news#articles1.csv.
def get_all_the_news() -> List[str]:
    data = pd.read_csv(f"{DATASET_PATH}articles1.csv", nrows=MAX_DOCS)
    return data['content'].tolist()


def get_20newsgroups() -> List[str]:
    return fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes')).data[:MAX_DOCS]
