from fastapi import APIRouter
from app.api.v1.endpoints import router as endpoints_router
from app.api.v1.nlp import router as nlp_router

api_v1_router = APIRouter()
api_v1_router.include_router(endpoints_router)
api_v1_router.include_router(nlp_router)

