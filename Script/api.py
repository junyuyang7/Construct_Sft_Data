import argparse
import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from typing import List, Literal

async def document():
    return RedirectResponse(url='/docs')

def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get()
