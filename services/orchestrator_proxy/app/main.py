from fastapi import FastAPI
from app.routers import openai_facade, context_managenent
from libs.utils.logger import setup_loggers

app = FastAPI(debug=True)

app.include_router(openai_facade.router, prefix="/v1", tags=["openai"])
app.include_router(context_managenent.router, prefix="/v1", tags=["management"])
setup_loggers()