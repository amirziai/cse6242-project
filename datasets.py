import csv
import pandas as pd
from typing import List

DATASETS_PATH = 'resources/datasets/'


def get_bbc() -> List[str]:
    return [line[1] for i, line in enumerate(csv.reader(open(f"{DATASETS_PATH}bbc-text.csv"), delimiter=',')) if i > 0]


# articles1.csv from https://www.kaggle.com/snapcrack/all-the-news#articles1.csv.
def get_all_the_news() -> List[str]:
    data = pd.read_csv(f"{DATASETS_PATH}articles1.csv", nrows=2000)
    return data['content'].tolist()
