
from imp import reload
from typing import List, Dict, Union

from fastapi.responses import Response, RedirectResponse
from fastapi import FastAPI
from api.v0 import api_v0
from utils import load_config, set_environment_variables

import uvicorn
import logging

logging.basicConfig(filename='logs.log', format='%(asctime)s - %(pathname)s - %(funcName)s - Line: %(lineno)d - %(levelname)s - %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', filemode='a', level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(api_v0.router)

@app.get("/")
async def read_root():
    """
    Root endpoint, redirect to most recent version of api.
    """
    return Response(content="Most recent api version: /api/v0", status_code=200)

@app.get("/info")
async def get_info():
    """
    Root info endpoint, redirect to docs.
    """
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    load_config()
    set_environment_variables()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)