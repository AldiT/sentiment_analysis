
from pathlib import Path
from fastapi.responses import Response, JSONResponse
from pydantic import JsonError

from utils import load_config




class DataController:

    def __init__(self, database_path: Path = Path("./dataset_database.yaml")):
        self.database = load_config(database_path)
    

    def get_data_list(self):
        return JSONResponse(content=self.database, status_code=200)