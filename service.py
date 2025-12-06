from ultralytics import YOLO
from PIL import Image
import cv2
import easyocr
import os

model_lisence_plates = YOLO('models/yolo_lisence_plate.pt')
model_vehicle = YOLO('models/yolov8m.pt')
vehicles = [2, 3, 5, 7]
reader = easyocr.Reader(["en"], gpu=True)

HOME = os.getcwd()  # Getting the current working directory

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


def check_image(file_name, images_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        file_path = os.path.join(images_folder, file_name)
        return file_path
    return None


def post_proccesing_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def save_image(image):
    cv2.imwrite("cache.png", image)

def recognition_vehicles(images):
    crop_vehicle_images = []
    for image in images:
        # Run the YOLO model on the current image
        results_vehicles = model_vehicle(image)[0]

        for detection in results_vehicles.boxes.data.tolist():
            carx1, cary1, carx2, cary2, conf, cls_cars = detection[:6]

            if int(cls_cars) in vehicles and conf >= 0.80:
                crop_vehicle_images.append(image[int(cary1):int(cary2), int(carx1):int(carx2)])
    return crop_vehicle_images


def ocr_detections(lisence_crop_img):
    lisence_detection = reader.readtext(lisence_crop_img)
    text = ""
    for lisence_text in lisence_detection:
        text += lisence_text[1]
    return text


def recognition_lisence_plate(images):
    detections = []
    for vehicle_crop_img in images:
        # Run the YOLO model on the current vehicle image
        results_lisence = model_lisence_plates(vehicle_crop_img)[0]

        for lisence in results_lisence.boxes.data.tolist():
            x1, y1, x2, y2, conf = lisence[:5]

            if conf < 0.7:
                continue

            lisence_crop_img = vehicle_crop_img[int(y1):int(y2), int(x1):int(x2)] 
            lisence_crop_img = post_proccesing_gray_image(lisence_crop_img)

            save_image(lisence_crop_img)
            detections.append([ocr_detections(lisence_crop_img), (x1, x2, y1, y2)])
    
            cv2.imshow('cropped', lisence_crop_img)
            cv2.waitKey(0)

    return detections


def detect_lisence_plates(images_folder):
    detections = []
    images = []
    for file_name in os.listdir(images_folder):

        # Check images that we get
        file_path = check_image(file_name=file_name, images_folder=images_folder)
        if type(file_path) is None:
            continue

        image = cv2.imread(file_path)
        images.append(image)

    detections.extend(recognition_lisence_plate(recognition_vehicles(images=images)))
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
    category = "images"
    images_folder = f"{HOME}/{category}"
    detections = detect_lisence_plates(images_folder=images_folder)
    print(*detections, sep="\n")


if __name__ == "__main__":
    main()