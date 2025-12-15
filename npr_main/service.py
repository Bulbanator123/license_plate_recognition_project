from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import cv2
from PIL import Image
from io import BytesIO
import tempfile
import os
import json
from fastapi.responses import JSONResponse, Response, FileResponse
from model import detect_nlpr_by_image, detect_nlpr_by_video, draw_buety_detections_on_image, draw_buety_detections_on_video


app = FastAPI()


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
            
        elif file.filename.endswith((".mp4", ".avi")):
            video = cv2.VideoCapture(tmp_path)
            detections = detect_nlpr_by_video(video)
        os.unlink(tmp.name)
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

    # read image with OpenCV
    if file.filename.endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(tmp_path)

        new_img = draw_buety_detections_on_image(img, det_dict)
        # 2. Конвертируем BGR (OpenCV) в RGB (обычный формат)
        image_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image_rgb)
        
        # Сохраняем в BytesIO
        img_io = BytesIO()
        pil_image.save(img_io, 'PNG')
        img_io.seek(0)
        os.unlink(tmp_path)

        # return image
        return Response(
            content=img_io.getvalue(), 
            media_type="image/png", 
            status_code=200)
    
    elif file.filename.endswith((".mp4")):
        video = cv2.VideoCapture(tmp_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_output:
            output_path = temp_output.name
        
        draw_buety_detections_on_video(output_path, video, det_dict)

        # Удаляем временные файлы
        os.unlink(tmp_path)
        
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename="result.mp4"
        )