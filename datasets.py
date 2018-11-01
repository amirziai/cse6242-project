import csv
from typing import List

DATASETS_PATH = 'resources/datasets/'


def get_bbc() -> List[str]:
    return [line[1] for i, line in enumerate(csv.reader(open(f"{DATASETS_PATH}bbc-text.csv"), delimiter=',')) if i > 0]
