from fastapi import APIRouter
from api.v0.models import routes

router = APIRouter(prefix="/api/v0")
router.include_router(routes.router)