
from typing import List, Dict, Union

from fastapi.responses import Response, RedirectResponse
from fastapi import FastAPI
from api.v0 import api_v0

app = FastAPI()

app.include_router(api_v0.router)

@app.get("/")
async def read_root():
    return Response(content="Most recent api version: /api/v0", status_code=200)

@app.get("/info")
async def get_info():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    print("Running the api.py as a main file.")