from ultralytics import YOLO
import cv2
from fast_plate_ocr import LicensePlateRecognizer
import os
import numpy as np


model_lisence_plates = YOLO('model/yolo_lisence_plate.pt')
model_vehicle = YOLO('model/yolov8m.pt')
vehicles = [2, 3, 5, 7]

reader = LicensePlateRecognizer("cct-xs-v1-global-model")

HOME = os.getcwd()  # Getting the current working directory
category_img = "images"
category_save = "saves"


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

        if int(cls_cars) in vehicles and conf >= 0.70:
            crop_vehicle_images.append([image[int(cary1):int(cary2), int(carx1):int(carx2)], (carx1, cary1, carx2, cary2)])
    return crop_vehicle_images


def ocr_detections(lisence_crop_img):
    lisence_detection, score = reader.run(lisence_crop_img, return_confidence=True)
    if sum(score[0]) / len(score[0]) > 0.70:
        return lisence_detection
    return None


def recognition_lisence_plate(data: list):
    detections = []
    for vehicle_crop_img, carcoords in data:
            # Run the YOLO model on the current vehicle image
        results_lisence = model_lisence_plates(vehicle_crop_img)[0]

        # cv2.imshow('original video', results_lisence.plot())
        # cv2.waitKey(0)
        
        for lisence in results_lisence.boxes.data.tolist():
            x1, y1, x2, y2, conf = lisence[:5]

            if conf < 0.70:
                continue
            lisence_crop_img = vehicle_crop_img[int(y1):int(y2), int(x1):int(x2)]
            # lisence_crop_img = post_proccesing_image(lisence_crop_img)
            # Now apply the OCR on the processed image
            det_text = ocr_detections(lisence_crop_img)
            if det_text is not None:
                detections.append({"lp_text": " ".join(det_text), "lp_coords": (x1 + carcoords[0], x2 + carcoords[0], y1 + carcoords[1], y2 + carcoords[1]), 
                                   "car_coords": (carcoords[0], carcoords[2], carcoords[1], carcoords[3])})
                save_lp(lisence_crop_img)
            # cv2.imshow('cropped', lisence_crop_img)
            # cv2.waitKey(0)

    return detections


def detect_lisence_plates_in_folder(images_folder):
    detections = []

    if type(images_folder) is not str: 
        return None
    
    for file_name in os.listdir(images_folder):
        file_path = os.path.join(images_folder, file_name)

        if file_name.endswith((".png", ".jpg", ".jpeg")):
            # Check images that we get

            image = cv2.imread(file_path)
            det = [detect_nlpr_by_image(image)]
            new_img = draw_buety_detections(image, det)
            cv2.imshow("buety_image", new_img)
            cv2.waitKey(0)
            detections.extend(det)

        if file_name.endswith((".mp4", ".mov", ".avi", ".webm", ".giff")):
            # Check videos that we get

            cap = cv2.VideoCapture(file_path)
            detections.extend(detect_nlpr_by_video(cap))
                
    return detections


def detect_nlpr_by_image(image):
    detections = {0: recognition_lisence_plate(recognition_vehicles(image=image))}
    return detections


def detect_nlpr_by_video(video):
    detections = []
    frame_num = -1              
    ret = True
    while ret:
        frame_num += 1
        ret, frame = video.read()
        if ret == True:
            print(frame_num)
            det = recognition_lisence_plate(recognition_vehicles(image=frame))
            if det:
                detections.append({frame_num: det})
    return detections


def draw_buety_detections(image, detections):
    # create copy of image
    img_copy = image.copy()
    
    # colors
    CAR_COLOR = (0, 255, 0)      # car
    LP_COLOR = (0, 0, 255)       # lp
    TEXT_COLOR = (255, 255, 255) # text
    TEXT_BG_COLOR = (0, 0, 0)    # text_back
    for frame in detections["detections"]:
        print(detections["detections"])
        for car in detections["detections"][frame]:
            print(car)
            # car coords
            car_x1, car_x2, car_y1, car_y2 = [int(coord) for coord in car["car_coords"]]
            
            # car rectangle
            cv2.rectangle(img_copy, 
                        (car_x1, car_y1), 
                        (car_x2, car_y2), 
                        CAR_COLOR, 2)
            
            # lp coords
            lp_x1, lp_x2, lp_y1, lp_y2 = [int(coord) for coord in car["lp_coords"]]
            
            # lp rectangle
            cv2.rectangle(img_copy, 
                        (lp_x1, lp_y1), 
                        (lp_x2, lp_y2), 
                        LP_COLOR, 3)
            
            # lp text
            lp_text = car.get("lp_text", "")
            
            # add text above car's bounding box
            if lp_text:
                # text size for lp text
                text_size = cv2.getTextSize(lp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                

                text_x = lp_x1
                text_y = lp_y1 - 10 if lp_y1 - 10 > 10 else lp_y2 + 25
                
                # draw background for lp text
                cv2.rectangle(img_copy,
                            (text_x, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 10, text_y + 5),
                            TEXT_BG_COLOR,
                            -1)
                
                # lp text
                cv2.putText(img_copy,
                        lp_text,
                        (text_x + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        TEXT_COLOR,
                        2)
            
            # add "Car" label
            cv2.putText(img_copy,
                    "Car",
                    (car_x1, car_y1 - 10 if car_y1 - 10 > 10 else car_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    CAR_COLOR,
                    2)
            
    return img_copy


def main():
    # Path to the images folder
    clear_folder()
    category_img = "images"
    images_folder = f"{HOME}/{category_img}"
    detections = {"detections": detect_lisence_plates_in_folder(images_folder=images_folder)}
    print(*detections, sep="\n")

    # image = cv2.imread("images/image0.png")
    # det = detect_nlpr_by_image(image)
    # print(det[0]["carcoords"])
    # print(det[0]["lp_coords"])
    # draw_rectangle(image=image, xy=det[0]["lp_coords"])


if __name__ == "__main__":
    main()