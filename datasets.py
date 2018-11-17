import csv
import sys
from typing import List

DATASETS_PATH = 'resources/datasets/'


csv.field_size_limit(sys.maxsize)

def get_bbc() -> List[str]:
    return [line[1] for i, line in enumerate(csv.reader(open(f"{DATASETS_PATH}bbc-text.csv"), delimiter=',')) if i > 0]

# articles1.csv from https://www.kaggle.com/snapcrack/all-the-news#articles1.csv.
def get_all_the_news() -> List[str]:
	return [line[9] for i, line in enumerate(csv.reader(open(f"{DATASETS_PATH}articles1.csv"), delimiter=',')) if i > 0 and i < 15000]