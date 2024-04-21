from pathlib import Path
import json


def load_jsons(data_p):
    with Path(data_p).open('r', encoding='utf-8') as r_f:
        data = json.load(r_f)
    return data