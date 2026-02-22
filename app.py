"""
AI Financial Agent — Web UI entry point.

Usage:
    python app.py
    uvicorn app:app --reload
"""

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="AI Financial Agent")

# Static assets (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes
from src.web.routes import router  # noqa: E402

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
