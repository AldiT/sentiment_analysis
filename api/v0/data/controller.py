
from pathlib import Path
from fastapi.responses import Response, JSONResponse, FileResponse
from typing import Dict
from api.v0 import data

from utils import load_config




class DataController:

    def __init__(self, database_path: Path = Path("./dataset_database.yaml")):
        self.database = load_config(database_path)
    

    def get_dataset_list(self) -> JSONResponse:
        """
        Method returns information about available datasets.

        Returns:
            Return a JsonReponse.
        """
        return JSONResponse(content=self.database, status_code=200)

    def get_dataset(self, dataset_id: int) -> JSONResponse:
        """
        Method returns information about a dataset given the id.
        Args:
            dataset_id: The unique id of the required dataset.
        Returns:
            Return a JsonReponse.
        """
        dataset_data = self._get_dataset_data(dataset_id)

        if dataset_data:
            return JSONResponse(content=dataset_data, status_code=200)
        return Response(content="Dataset not found!", status_code=404)
    
    def get_download_dataset(self, dataset_id: int) -> FileResponse:
        """
        Method returns a dataset given the id.
        Args:
            dataset_id: The unique id of the required dataset.
        Returns:
            Return a FileResponse.
        """
        dataset_data = self._get_dataset_data(dataset_id)

        if not dataset_data:
            return Response(content="Dataset not found!", status_code=404)
        
        file_path = Path(f"./{dataset_data['foldername']}/{dataset_data['filename']}")
        if file_path.exists():
            return FileResponse(path=file_path, status_code=200)
        else:
            return Response(content="Dataset not found!", status_code=404)
    
    def _get_dataset_data(self, dataset_id: int) -> Dict:
        """
        Method returns information about a dataset given the id.
        Args:
            dataset_id: The unique id of the required dataset.
        Returns:
            Return a dataset information.
        """
        dataset_data = None

        for dataset in self.database["datasets"]:
            if dataset["id"] == dataset_id:
                return dataset
        return None
