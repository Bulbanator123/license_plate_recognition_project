from ultralytics import YOLO
from PIL import Image
import cv2
import easyocr
import os

model_lisence_plates = YOLO('models/yolo_lisence_plate.pt')
model_vehicle = YOLO('models/yolov8m.pt')
vehicles = [2, 3, 5, 7]
reader = easyocr.Reader(["en"])

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

# Path to the images folder
category = "images"
images_folder = f"{HOME}/{category}"
target_classes = [1]

for file_name in os.listdir(images_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        file_path = os.path.join(images_folder, file_name)

        # Load the image
        image = cv2.imread(file_path)

        # Run the YOLO model on the current image
        results_vehicles = model_vehicle(file_path)[0]
        annotated_image = results_vehicles.plot() 
        cv2.imshow("YOLO Custom Detection", annotated_image)
        cv2.waitKey(0)

        # Extract bounding box coordinates and class names
        detections = []
        bordered_img = image
        for detection in results_vehicles.boxes.data.tolist():
            carx1, cary1, carx2, cary2, conf, cls_cars = detection[:6]

            if int(cls_cars) in vehicles and conf >= 0.80:
                vehicle_crop_img = image[int(cary1):int(cary2), int(carx1):int(carx2)] 
                results_lisence = model_lisence_plates(vehicle_crop_img)[0]
                # cv2.imshow('cropped', vehicle_crop_img)
                # cv2.waitKey(0)
                annotated_image = results_lisence.plot() 
                cv2.imshow("YOLO Custom Detection", annotated_image)
                cv2.waitKey(0)
                for lisence in results_lisence.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = lisence[:6]

                    lisence_crop_img = vehicle_crop_img[int(y1):int(y2), int(x1):int(x2)] 
                    license_crop_gray = cv2.cvtColor(lisence_crop_img, cv2.COLOR_BGR2GRAY)
                    # _, license_plate_crop_thresh = cv2.threshold(license_crop_gray, 60, 255, cv2.THRESH_BINARY_INV)

                    cv2.imwrite("cache.png", lisence_crop_img)

                    lisence_detection = reader.readtext(lisence_crop_img)
                    text = ""
                    for lisence_text in lisence_detection:
                        text += lisence_text[1]
                    print(text)
                    cv2.imshow('cropped', lisence_crop_img)
                    cv2.waitKey(0)

                    detections.append(text)

                    bordered_img = draw_border(bordered_img, (int(carx1), int(cary1)), (int(carx2), int(cary2)), (0, 255, 0), 10)
                    
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 5)

                    cv2.putText(bordered_img, text, (int((carx2 + carx1 - text_width) / 2), int(cary1 + (text_height))),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5)

        cv2.imshow("complete_img", bordered_img)
        cv2.waitKey(0)
        print(f"Detected objects for {file_name}: {detections}")
        # Perform OCR on detected regions
        # perform_ocr(image, detections)
cv2.destroyAllWindows()
print("Processing complete for all images!")