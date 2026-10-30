import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vision import ObstacleDetector
from navigation import NavigationAssistant

app = FastAPI(title="AI Visual Assistance for Navigation")

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

    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return {
            "warnings": [],
            "boxes": [],
            "error": "Invalid image"
        }

    # Return both warning messages and bounding-box coordinates.
    return detector.analyze_frame(frame, include_boxes=True)


@app.post("/voice-command")
def handle_voice_command(req: VoiceCommandRequest):
    # Keep existing navigation command handling.
    return nav_assistant.process_command(
        req.lat,
        req.lon,
        req.command
    )


@app.post("/update-gps")
def handle_gps_update(req: GPSUpdateRequest):
    # Keep existing GPS updates, turns and route recalculation.
    return nav_assistant.update_position(
        req.lat,
        req.lon,
        req.accuracy
    )
