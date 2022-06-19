
from fastapi import APIRouter
from typing import Union

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
    return controller.get_model(model_id)

@router.get("/{model_id}/infer")
async def get_prediction(model_id: int, text: str):
    return controller.get_prediction(model_id, text)

@router.get("/{model_id}")
def get_models(model_id: int):
    return f"Model with id {model_id} was successfully requested!"

