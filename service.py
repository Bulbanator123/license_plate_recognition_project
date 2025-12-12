from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import tempfile
import os
from fastapi.responses import JSONResponse
from model import detect_nlpr_by_image, detect_nlpr_by_video

# class ClientData(BaseModel):
    

app = FastAPI()


@app.post("/nplr")
async def nplr(file: UploadFile = File(...)):
    try:   
        suffix = os.path.splitext(file.filename)[1]  # Получаем расширение
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        # Читаем изображение с помощью OpenCV
        if file.filename.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(tmp_path)
            detections = detect_nlpr_by_image(img)
        elif file.filename.endswith((".mp4", ".mov", ".avi", ".webm", ".giff")):
            video = cv2.VideoCapture(tmp_path)
            detections = detect_nlpr_by_video(video)

        print(detections)
        return JSONResponse(
            content={
                "detections": detections
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

