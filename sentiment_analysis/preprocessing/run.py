from dask.distributed import Event, LocalCluster, Client
from sentiment_analysis.preprocessing.dask_utils import distribute
from sentiment_analysis.preprocessing.contractions import contractions
from pathlib import Path
from typing import List
from run import config

import pandas as pd
import click
import spacy
import asyncio
import logging
import sys
import re

from utils import save_cache_pkl

logger = logging.getLogger(__name__)

def read_data(path: Path, encoding: str = "ISO-8859-1"):
    if path.is_file():
        df = pd.read_csv(path, encoding=encoding, header=None)
        text, labels = list(df[3]), list(df[2])
        return text, labels
    else:
        logger.critical("Data file does not exist!")
        sys.exit(-1)

def remove_stopwords(texts: List[str], model = None) -> List[str]:
    if model is None:
        sys.exit(-1)
    result = []
    all_stopwords = model.Defaults.stop_words

    for doc in model.pipe(texts):
        result.append(" ".join([str(token) for token in doc if str(token) not in all_stopwords]))

    return result

def remove_special_chars(texts: List[str]) -> List[str]:
    result = []

    for i, text in enumerate(texts):
        if i % 5000 == 0:
            logger.info(f"Special char removal: {i}")

        if isinstance(text, str):
            for word in text.split(): # remove short forms: e.g. I'm --> I am
                if word.lower() in contractions:
                    text = text.replace(word, contractions[word.lower()])
            text = re.sub(r'\S*@\S*\s?', '', text) #remove email addresses
            text = re.sub(r'http\S+', '', text) # remove urls
            result.append(re.sub('[^A-Za-z0-9]+', ' ', text)) # keep only alphanumeric characters
        else:
            result.append(" ")

    return result

def clean(texts: List[List[str]], dask_event_name: str = None, spacy_model: str = "en_core_web_sm"):
    nlp = spacy.load(spacy_model)

    texts = remove_special_chars(texts)
    logger.info("Removed special characters.")
    texts = remove_stopwords(texts, model=nlp)
    logger.info("Removed stopwords.")

    if dask_event_name:
        event = Event(dask_event_name)
        event.set()
    return texts

def filter_nans(texts: List[str], labels: List[str]) -> List[str]:
    result_str, result_lbl = [], []

    for i, text in enumerate(texts):
        if isinstance(text, str):
            result_str.append(text)
            result_lbl.append(labels[i])
    
    return result_str, result_lbl

def run_sequential(save: bool = True):
    logger.info("Running preprocessing sequentially!")
    
    data_path = Path(config["ml"]["preprocessing"]["data_path"])
    save_to_folder = str(config["ml"]["preprocessing"]["save_to"]["folder"])
    save_to_filename = str(config["ml"]["preprocessing"]["save_to"]["filename"])
    num_workers = int(config["ml"]["preprocessing"]["num_workers"])
    cache = int(config["ml"]["preprocessing"]["cache"])


    if cache == 1 and Path(f"./cache/{save_to_folder}/{save_to_filename}").exists():
        logger.info("Result file already exists, skipping preprocessing!")
        return

    logger.info(f"Reading the data from: {data_path}")
    texts, labels = read_data(data_path)

    texts, labels = filter_nans(texts, labels)

    logger.info("Preprocessing starting...")

    texts = clean(texts)

    if save:
        logger.info("Saving results!")
        save_cache_pkl((texts, labels), save_to_folder, save_to_filename)
    
    logger.info("Preprocessing done!")




async def run_distribute():
    logger.info("Running preprocessing distributed!")

    data_path = Path(config["ml"]["preprocessing"]["data_path"])
    save_to_folder = str(config["ml"]["preprocessing"]["save_to"]["folder"])
    save_to_filename = str(config["ml"]["preprocessing"]["save_to"]["filename"])
    num_workers = int(config["ml"]["preprocessing"]["num_workers"])
    cache = int(config["ml"]["preprocessing"]["cache"])

    

    local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, processes=True)
    client = await Client(local_cluster, asynchronous=True)

    if cache == 1 and Path(f"./cache/{save_to_folder}/{save_to_filename}").exists():
        logger.info("Result file already exists, skipping preprocessing!")
        client.close()
        return
        
    logger.info(f"Reading the data from: {data_path}")
    texts, labels = read_data(data_path)

    texts, labels = filter_nans(texts, labels)

    logger.info("Preprocessing starting...")
    func_params = {
        "datasets": texts,
    }
    texts = await distribute(client, clean, func_params, num_workers=num_workers)

    logger.info("Saving results!")
    save_cache_pkl((texts, labels), save_to_folder, save_to_filename)
    
    logger.info("Preprocessing done!")

    client.close()

@click.command()
@click.option('--num-workers', default=16, help="The number of workers that will be used for preprocessing.")
def run(num_workers: int):
    distribute: bool = bool(config["ml"]["preprocessing"]["distribute"])

    if distribute:
        asyncio.get_event_loop().run_until_complete(run_distribute())
    else:
        run_sequential()



if __name__ == "__main__":
    run()