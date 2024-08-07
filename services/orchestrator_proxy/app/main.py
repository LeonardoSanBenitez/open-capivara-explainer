from fastapi import FastAPI
from app.routers import openai_facade, context_managenent
from libs.utils import setup_loggers

app = FastAPI(debug=True)

app.include_router(openai_facade.router, prefix="/openai", tags=["openai"])
app.include_router(context_managenent.router, prefix="/openai", tags=["management"])
setup_loggers()