
from fastapi import APIRouter
from typing import Union

router = APIRouter(
    prefix="/models"
)

@router.get("/")
def get_models():
    return {"models": [
        {"id": 1, "name": "lstm"},
        {"id": 2, "name": "transformer"}
    ]}


@router.get("/{model_id}")
def get_models_with_text(model_id: int, text: str):
    return f"Model with id {model_id} was passed text {text}!"


@router.get("/{model_id}")
def get_models(model_id: int):
    return f"Model with id {model_id} was successfully requested!"

