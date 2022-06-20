
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from typing import Union, List

from api.v0.models.controller import ModelController

router = APIRouter(
    prefix="/models"
)

controller = ModelController()

@router.get("/")
async def get_models():
    return controller.get_models()

@router.get("/{model_id}")
async def get_model(model_id: int) -> JSONResponse:
    """
    This endpoint gets a model id and returns information about that model.
    """
    return controller.get_model_info(model_id)

@router.get("/{model_id}/download")
async def get_model(model_id: int) -> FileResponse:
    """
    This endpoint gets a model id and returns the model file(s).
    """
    return controller.get_model(model_id)

@router.get("/{model_id}/infer")
async def get_prediction(model_id: int, text: str) -> JSONResponse:
    """
    This endpoint gets a text and returns a sentiment prediction for it.
    """
    return controller.get_prediction(model_id, text)

@router.post("/{model_id}/batch")
async def get_batch_prediction(model_id: int, request: Request):
    body = await request.json()
    return controller.get_batch_prediction(model_id, body["texts"])

@router.post("/{model_id}/batch/csv")
async def get_batch_prediction(model_id: int, csv_file: UploadFile):
    """
    This endpoint gets a csv file containing texts and returns the sentiment prediction results.
    @TODO: Still not impolemented.
    """
    return Response(content="This method is not yet implemented, sorry for the inconvenience", status_code=200)#controller.get_batch_prediction(model_id, csv_file)


