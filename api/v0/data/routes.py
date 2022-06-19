
from fastapi import APIRouter
from fastapi.responses import Response, JSONResponse, FileResponse
from typing import Union, List

from api.v0.data.controller import DataController

import pandas as pd
import os
import sys

router = APIRouter(
    prefix="/data"
)

controller = DataController()


@router.get("/")
async def get_data_list():
    return controller.get_data_list()
