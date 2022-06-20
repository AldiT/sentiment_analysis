from fastapi import APIRouter
from api.v0.models import routes as model_routes
from api.v0.data import routes as data_routes

router = APIRouter(prefix="/api/v0")

router.include_router(model_routes.router)
router.include_router(data_routes.router)