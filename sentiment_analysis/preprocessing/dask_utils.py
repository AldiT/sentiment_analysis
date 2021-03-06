from typing import Dict, List
from dask.distributed import Event, Client, LocalCluster, get_client
from copy import copy

import logging
logger = logging.getLogger(__name__)


def split_list_data(data: List, splits: int = 16):
    """
    A generator that yields the datasets in as many splits as specified for distribution purposes.
    Args:
        data: The datasets to be split in a List.
        splits: The number of splits to be generated from the datasets.

    Yields:
        Splits of datasets every run.
    """
    logger.info("Spliting datasets!")
    chunk_size = len(data) // splits
    i = 0
    for i in range(0, len(data), chunk_size):
        to_index = i + chunk_size if i + chunk_size < len(data) else len(data)
        yield data[i: to_index]

async def get_client_local(worker_cluster: LocalCluster) -> Client:
    """
    A helper function to retrieve the global client object or create a new one if it doesn't exist.
    Args:
        worker_cluster: The cluster of workers which the client object will manage (if a new one is created).

    Returns:
        Splits of datasets every run.
    """
    try:
        return await get_client()
    except ValueError:
        logger.info("Existing client not found, creating new one!")
        return await Client(worker_cluster, asynchrounous=True)

async def distribute(client: Client, func, func_params: Dict = None, num_workers: int = 16):
    """
    A helper function to launch a distributed job on the client object.
    Args:
        client: The client on the underlying cluster of which the job should be run.
        func: A function (job) to be distributed.
        func_params: The parameters to be passed to the function (job).
        num_workers: The number of workers to be used for this job.
    Returns:
        Returns the accumulated result from all the workers.
    """
    events = []
    futures = []
    data_futures = []
    params = copy(func_params)
    del params["datasets"]

    for split_number, split_chunk in enumerate(split_list_data(func_params['datasets'], splits=num_workers)):
        event_name = f"event_{split_number}"
        events.append(Event(event_name))
        if func_params:
            futures.append(client.submit(func, split_chunk, dask_event_name=event_name, **params))
        else:
            futures.append(client.submit(func, split_chunk, dask_event_name=event_name))

    for event in events:
        logger.debug(f"Waiting for event {event}")
        event.wait()
        logger.debug(f"Waiting for event {event} done!")

    result = []
    for future_index, future in enumerate(futures):
        logger.debug(f"Getting results for future number: {future_index}.")
        res = future.result()
        if len(res) != 0:
            result.extend(res)
        logger.debug(f"Result for future {future_index} retrieved successfully!")
    del futures

    logger.info("Distributed computing done!")
    return result