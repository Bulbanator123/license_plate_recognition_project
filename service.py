from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
import cv2
from pydantic import BaseModel, Field
from typing import Dict, List, Any
from PIL import Image
from io import BytesIO
import tempfile
import os
import json
from fastapi.responses import JSONResponse, Response
from model import detect_nlpr_by_image, detect_nlpr_by_video, draw_buety_detections


app = FastAPI()

# Модель для внутреннего объекта
class DetectionItem(BaseModel):
    lp_text: str
    lp_coords: List[float] = Field(...)
    car_coords: List[float] = Field(...)

# Модель для запроса
class DetectionRequest(BaseModel):
    detections: Dict[int, List[DetectionItem]]


@app.post("/nplr")
async def nplr(file: UploadFile = File(...)):
    try:   
        suffix = os.path.splitext(file.filename)[1]  # read ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        # read image with OpenCV
        if file.filename.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(tmp_path)
            detections = detect_nlpr_by_image(img)
            
        elif file.filename.endswith((".mp4", ".mov", ".avi", ".webm", ".giff")):
            video = cv2.VideoCapture(tmp_path)
            detections = detect_nlpr_by_video(video)

        return JSONResponse(
            content={
                "detections": detections
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)


@app.post("/nplr/image")
async def return_nplr_image(data: str = Form(...), file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]  # read ext
    det_dict = json.loads(data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    img = cv2.imread(tmp_path)
    print(det_dict)
    new_img = draw_buety_detections(img, det_dict)
    cv2.imshow("weriw", new_img)
    cv2.waitKey(0)
    # 2. Конвертируем BGR (OpenCV) в RGB (обычный формат)
    image_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image_rgb)
    
    # Сохраняем в BytesIO
    img_io = BytesIO()
    pil_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Возвращаем изображение
    # return JSONResponse(content=det_dict)
    return Response(
        content=img_io.getvalue(), 
        media_type="image/png", 
        status_code=200)