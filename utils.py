from ast import Load
from pathlib import Path
from dotenv import load_dotenv
from keras import backend as K

import pickle as pkl
import logging
import yaml
import os
import sys

logging.basicConfig(filename='logs.log', format='%(asctime)s - %(pathname)s - %(funcName)s - Line: %(lineno)d - %(levelname)s - %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', filemode='a', level=logging.INFO)

logger = logging.getLogger(__name__)



def load_config(path: Path = Path("./config.yaml")):
    if not path.is_file():
        sys.exit(-1)

    with open(path, "r") as handler:
        config = yaml.load(handler, Loader=yaml.FullLoader)
        logger.info("Config file read successfully!")

    return config

def set_environment_variables():
    load_dotenv()

def save_cache_pkl(obj: object, folder_name: str, file_name: str):
    path = Path(f"./cache/{folder_name}/{file_name}")
    try:
        with path.open("wb") as handle:
            pkl.dump(obj, handle)
    except FileNotFoundError as fnf:
        logger.error(f"File: {path} not found in cache.")
        sys.exit(-1)
    except Exception as e:
        logger.error(e)
        sys.exit(-1)

def load_cache_pkl(folder_name: str, file_name: str) -> object:
    path = Path(f"./cache/{folder_name}/{file_name}")

    try:
        with path.open("rb") as handle:
            obj = pkl.load(handle)
    except FileNotFoundError as fnf:
        logger.error(f"File: {path} not found in cache.")
        sys.exit(-1)
    except Exception as e:
        logger.error(e)
        sys.exit(-1)

    return obj

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == "__main__":
    load_config()
    set_environment_variables()

