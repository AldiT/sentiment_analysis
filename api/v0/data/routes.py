
from fastapi import APIRouter
from fastapi.responses import Response, JSONResponse, FileResponse
from typing import Union, List

from api.v0.data.controller import DataController

import pandas as pd
import os
import sys

router = APIRouter(
    prefix="/datasets"
)

controller = DataController()


@router.get("/")
async def get_dataset_list():
    return controller.get_dataset_list()

@router.get("/{dataset_id}")
async def get_dataset(dataset_id: int):
    controller.get_dataset(dataset_id)

@router.get("/{dataset_id}/download")
async def get_download_dataset(dataset_id: int):
    """
        Endpoint description.
    """
    controller.get_download_dataset(dataset_id)
