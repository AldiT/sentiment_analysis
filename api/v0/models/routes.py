
from fastapi import APIRouter, File, UploadFile, Request
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
async def get_model(model_id: int):
    return controller.get_model_info(model_id)

@router.get("/{model_id}/download")
async def get_model(model_id: int):
    return controller.get_model(model_id)

@router.get("/{model_id}/infer")
async def get_prediction(model_id: int, text: str):
    return controller.get_prediction(model_id, text)

@router.post("/{model_id}/batch")
async def get_batch_prediction(model_id: int, request: Request):
    body = await request.json()
    return controller.get_batch_prediction(model_id, body["texts"])

@router.post("/{model_id}/batch/csv")
async def get_batch_prediction(model_id: int, csv_file: UploadFile):
    return controller.get_batch_prediction(model_id, csv_file)

@router.get("/{model_id}")
def get_models(model_id: int):
    return f"Model with id {model_id} was successfully requested!"

