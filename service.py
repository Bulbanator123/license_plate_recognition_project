from ultralytics import YOLO
from PIL import Image
import cv2
import easyocr
import os
import numpy as np


model_lisence_plates = YOLO('model/yolo_lisence_plate.pt')
model_vehicle = YOLO('model/yolov8m.pt')
vehicles = [2, 3, 5, 7]

reader = easyocr.Reader(["en"], gpu=True)

HOME = os.getcwd()  # Getting the current working directory
category_img = "images"
category_save = "saves"


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def post_proccesing_image(image):
    thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return thresh


def get_free_filename():
    i = 1
    while i < 1000000000:
        num = str(i).rjust(8, "0")
        filename = f"cache{num}.png"
        path = os.path.join(category_save, filename)

        if not os.path.exists(path):
            return filename

        i += 1
    raise Exception("DataBase is overfilled")


def save_lp(image):
    filename = get_free_filename()
    cv2.imwrite(f"{category_save}/{filename}", image)


def clear_folder():
    all_items = os.listdir(category_save)
    for path in all_items:
        file_to_delete = os.path.join(category_save, path)
        os.remove(file_to_delete)


def recognition_vehicles(image):
    crop_vehicle_images = []
    # Run the YOLO model on the current image
    results_vehicles = model_vehicle(image)[0]

    for detection in results_vehicles.boxes.data.tolist():
        carx1, cary1, carx2, cary2, conf, cls_cars = detection[:6]

        if int(cls_cars) in vehicles and conf >= 0.80:
            crop_vehicle_images.append(image[int(cary1):int(cary2), int(carx1):int(carx2)])
    return crop_vehicle_images


def ocr_detections(lisence_crop_img):
    text = ""
    lisence_detection = reader.readtext(lisence_crop_img, width_ths=0.1)
    for lisence_text in lisence_detection:
        text += lisence_text[1]
    return text


def recognition_lisence_plate(images):
    for vehicle_crop_img in images:
        detections = []
            # Run the YOLO model on the current vehicle image
        results_lisence = model_lisence_plates(vehicle_crop_img)[0]

        # cv2.imshow('original video', results_lisence.plot())
        # cv2.waitKey(0)
        
        for lisence in results_lisence.boxes.data.tolist():
            x1, y1, x2, y2, conf = lisence[:5]

            if conf < 0.7:
                continue
            lisence_crop_img = vehicle_crop_img[int(y1):int(y2), int(x1):int(x2)] 
            lisence_crop_img = post_proccesing_image(lisence_crop_img)

            # Now apply the OCR on the processed image
            det_text = ocr_detections(lisence_crop_img)
            if len(det_text) > 0:
                detections.append([det_text, (x1, x2, y1, y2)])
                save_lp(lisence_crop_img)
            # cv2.imshow('cropped', lisence_crop_img)
            # cv2.waitKey(0)

    return detections


def detect_lisence_plates_in_folder(images_folder):
    detections = []
    images = []

    if type(images_folder) is not str: 
        return None
    
    for file_name in os.listdir(images_folder):
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            # Check images that we get
            file_path = os.path.join(images_folder, file_name)
            if type(file_path) is None:
                continue

            image = cv2.imread(file_path)
            detections.extend([recognition_lisence_plate(recognition_vehicles(image=image))])

        if file_name.endswith((".mp4", ".mov", ".avi", ".webm", ".giff")):
            # Check videos that we get
            file_path = os.path.join(images_folder, file_name)
            if type(file_path) is None:
                continue

            frame_num = -1
            ret = True
            cap = cv2.VideoCapture('videos/sample.mp4')
            while ret:
                frame_num += 1
                ret, frame = cap.read()
                if ret == True:
                    # frame = cv2.resize(frame, (780, 540), interpolation=cv2.INTER_LINEAR)
                    print(frame_num)
                    detections.extend(recognition_lisence_plate(recognition_vehicles(image=frame)))
                
    return detections


# def completed_img():
#     bordered_img = draw_border(bordered_img, (int(carx1), int(cary1)), (int(carx2), int(cary2)), (0, 255, 0), 10)
            
#     (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 5)

#     cv2.putText(bordered_img, text, (int((carx2 + carx1 - text_width) / 2), int(cary1 + (text_height))),
#         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)
#     cv2.imshow("complete_img", bordered_img)
#     cv2.waitKey(0)


def main():
    # Path to the images folder
    clear_folder()
    category_img = "images"
    images_folder = f"{HOME}/{category_img}"
    detections = detect_lisence_plates_in_folder(images_folder=images_folder)
    print(*detections, sep="\n")


if __name__ == "__main__":
    main()