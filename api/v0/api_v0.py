from fastapi import APIRouter
from api.v0.routes import models

router = APIRouter(prefix="/api/v0")
router.include_router(models.router)