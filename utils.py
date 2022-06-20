from ast import Load
from pathlib import Path
from dotenv import load_dotenv
from keras import backend as K
import tensorflow as tf

import pickle as pkl
import logging
import yaml
import os
import sys

logger = logging.getLogger(__name__)



def load_config(path: Path = Path("./config.yaml")):
    """
    Load app config from a yaml file.
    Args:
        path: The path to the yaml file.
    Returns:
        Returns a python dictionary with all the information.
    """
    if not path.is_file():
        sys.exit(-1)

    with open(path, "r") as handler:
        config = yaml.load(handler, Loader=yaml.FullLoader)
        logger.info("Config file read successfully!")

    return config

def set_environment_variables():
    """
    Load environment variables for secrets like database names and passwords.
    """
    load_dotenv()

def save_cache_pkl(obj: object, folder_name: str, file_name: str):
    """
    Save a given object in the cache.
    Args:
        obj: The object to be stored in the cache.
        folder_name: The name of the folder inside the cache where the object should be stored.
        file_name: The name of the file where the object will be stored.
    Returns:
    """
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
    """
    Load object from the cache.
    Args:
        folder_name: The folder inside the cache where the file is located.
        file_name: The file name where the object binary is stored.
    Returns:
        Returns the retrieved object.
    """
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

def recall_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.float32:
    """
    A machine learning performance metric (Recall)
    Args:
        y_true: The array with the labels.
        y_pred: The array with the predicted scores.
    Returns:
        Returns the computed metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred) -> tf.float32:
    """
    A machine learning performance metric (Precision)
    Args:
        y_true: The array with the labels.
        y_pred: The array with the predicted scores.
    Returns:
        Returns the computed metric.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    """
    A machine learning performance metric (F1-score)
    Args:
        y_true: The array with the labels.
        y_pred: The array with the predicted scores.
    Returns:
        Returns the computed metric.
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

