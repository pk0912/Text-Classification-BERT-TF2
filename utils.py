"""
Python file containing methods that are common requirement throughout the project
"""

import logging
import logging.handlers as lh
from urllib import request
import pandas as pd
from joblib import dump
from settings import (
    LUCKY_SEED,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_BACKUP_COUNT,
)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logging(metaclass=Singleton):
    def __init__(self):
        self.logging = logging.getLogger("Sentiment Analysis")
        log_handler = lh.RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUP_COUNT
        )
        formatter = logging.Formatter(LOG_FORMAT)
        log_handler.setFormatter(formatter)
        self.logging.addHandler(log_handler)
        self.logging.setLevel(LOG_LEVEL)
        self.logging.debug("Initialising Logging!!!")


logger = Logging().logging


def download_and_write_to_file(url, filepath):
    try:
        response = request.urlopen(url)
        content = response.read()
        with open(filepath, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(
            "Exception in reading from url and writing locally : {}".format(str(e))
        )


def read_csv_data(path, keep_columns=[], drop_cols=[]):
    try:
        data = pd.read_csv(path, encoding="utf-8")
        if len(keep_columns) > 0:
            data = data[keep_columns]
        data = data.dropna(subset=drop_cols, how="all")
        data = data.sample(frac=1, random_state=LUCKY_SEED).reset_index(drop=True)
        return data
    except Exception as e:
        logger.error("Exception in reading csv data : {}".format(str(e)))


def save_csv_data(dataframe, path, index=False, encoding="utf-8"):
    dataframe.to_csv(path, index=index, encoding=encoding)


def save_objects(obj, path):
    try:
        dump(obj, path)
    except Exception as e:
        logger.error("ERROR IN SAVING OBJECT : {}".format(str(e)))
        return False
    return True


def save_model_summary(model, file_path):
    with open(file_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
