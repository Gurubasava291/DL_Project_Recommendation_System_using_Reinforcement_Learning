from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import torch
import random
from src.dqn_agent import DQNAgent

app = FastAPI(title="MLOps Recommender API")

# Mount static files for frontend
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize models (in a real scenario, load from MLflow registry or saved weights)
NUM_SONGS = 200
STATE_LENGTH = 3
dqn_agent = DQNAgent(num_songs=NUM_SONGS, state_length=STATE_LENGTH)

class RecommendRequest(BaseModel):
    state: list[int]
    k: int = 5
    model_type: str = "dqn" # 'baseline', 'dqn', 'qml'

@app.get("/")
async def read_index():
    return FileResponse("app/static/index.html")

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    if len(req.state) != STATE_LENGTH:
        raise HTTPException(status_code=400, detail=f"State length must be {STATE_LENGTH}")
    
    if req.model_type == "dqn":
        recs = dqn_agent.recommend(req.state, k=req.k)
        return {"recommendations": recs, "model": "Deep Q-Network"}
    elif req.model_type == "baseline":
        # Mocking popularity for now, ideally loads PopularityBaseline
        recs = random.sample(range(NUM_SONGS), req.k)
        return {"recommendations": recs, "model": "Popularity Baseline"}
    elif req.model_type == "qml":
        # Mocking QML recommendations
        recs = random.sample(range(NUM_SONGS), req.k)
        return {"recommendations": recs, "model": "Quantum Variational Classifier"}
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")

@app.get("/metrics")
async def get_metrics():
    # In a real app, query MLflow for latest metrics
    return {
        "latest_run": {
            "model": "DQN",
            "loss": 0.045,
            "reward_epoch": 120,
            "status": "Production"
        }
    }
