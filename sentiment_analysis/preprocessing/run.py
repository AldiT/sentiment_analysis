from dask.distributed import Event, LocalCluster, Client
from sentiment_analysis.preprocessing.dask_utils import distribute
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

    for text in texts:
        #text = re.sub(r"[üÜ]", "ue", text)
        #text = re.sub(r"[äÄ]", "ae", text)
        #text = re.sub(r"[öÖ]", "oe", text)
        #text = re.sub(r"ß", "ss", text)
        #result.append(" ".join(re.findall('[\w]+', text)))
        result.append(re.sub('[^A-Za-z0-9üÜäAöÖß]+', ' ', text))

    return result

def clean(texts: List[List[str]], dask_event_name: str = None, spacy_model: str = "en_core_web_sm"):
    nlp = spacy.load(spacy_model)
    result = []
    for text in texts:
        if isinstance(text, str):
            text = remove_special_chars(text)
            text = remove_stopwords(text, model=nlp)
            result.append(text)

    if dask_event_name:
        event = Event(dask_event_name)
        event.set()
    return result


async def run_distribute():
    data_path = Path(config["ml"]["preprocessing"]["data_path"])
    save_to = Path(config["ml"]["preprocessing"]["save_results_to"])
    num_workers = int(config["ml"]["preprocessing"]["num_workers"])
    cache = int(config["ml"]["preprocessing"]["cache"])

    if cache == 1 and save_to.exists():
        logger.info("Result file already exists, skipping preprocessing!")
        client.close()
        return

    local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, processes=True)
    client = await Client(local_cluster, asynchronous=True)
    
    logger.info(f"Reading the data from: {data_path}")
    texts = read_data(data_path)


    logger.info("Preprocessing starting...")
    func_params = {
        "datasets": texts,
    }
    texts = await distribute(client, clean, func_params, num_workers=num_workers)

    logger.info("Saving results!")
    save_cache_pkl(texts, "preprocessing", "initial_texts.pkl")
    
    logger.info("Preprocessing done!")

    client.close()

@click.command()
@click.option('--num-workers', default=16, help="The number of workers that will be used for preprocessing.")
def run(num_workers: int):
    asyncio.get_event_loop().run_until_complete(run_distribute())
    print("Running the run function.")



if __name__ == "__main__":
    run()