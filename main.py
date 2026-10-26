import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vision import ObstacleDetector
from navigation import NavigationAssistant

app = FastAPI()

detector = ObstacleDetector()
nav_assistant = NavigationAssistant()


class VoiceCommandRequest(BaseModel):
    lat: float
    lon: float
    command: str


class GPSUpdateRequest(BaseModel):
    lat: float
    lon: float
    accuracy: float = 10.0


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/analyze-vision")
async def analyze_vision(image_file: UploadFile = File(...)):
    contents = await image_file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    return {"warnings": detector.analyze_frame(frame)}


@app.post("/voice-command")
def handle_voice_command(req: VoiceCommandRequest):
    # Processes spoken commands ("take me home", "repeat", "stop")
    return nav_assistant.process_command(req.lat, req.lon, req.command)


@app.post("/update-gps")
def handle_gps_update(req: GPSUpdateRequest):
    # Checks live walking progress, turns, and off-route recalculations
    return nav_assistant.update_position(req.lat, req.lon, req.accuracy)