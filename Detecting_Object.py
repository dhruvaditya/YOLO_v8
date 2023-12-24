from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import math

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "telephone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

object_dimensions = {
    "bird": "0.10",
    "cat": "0.45",
    "backpack": "0.55",
    "umbrella": "0.50",
    "bottle": "0.20",
    "wine glass": "0.25",
    "cup": "0.15",
    "fork": "0.15",
    "knife": "0.25",
    "spoon": "0.15",
    "banana": "0.20",
    "apple": "0.07",
    "sandwich": "0.20",
    "orange": "0.08",
    "chair": "0.50",
    "laptop": "0.40",
    "mouse": "0.10",
    "remote": "0.20",
    "keyboard": "0.30",
    "phone": "0.14",
    "book": "0.18",
    "toothbrush": "0.16"
}

def voice_notification(obj_name, direction, distance):
    engine = pyttsx3.init()
    text = f"{obj_name.capitalize()} is at {direction}. It is {distance:.2f} meters away."
    engine.say(text)
    engine.runAndWait()

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    center_x = int(frame_width // 2)
    center_y = int(frame_height // 2)
    radius = min(center_x, center_y) - 30

    while True:
        success, img = cap.read()

        # Predict objects using the YOLO model
        results = model.predict(img, stream=True)

        # Draw clock
        for i in range(1, 13):
            angle = math.radians(360 / 12 * i - 90)
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))

            thickness = 3 if i % 3 == 0 else 1
            length = 20 if i % 3 == 0 else 10

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(i), (x - 10, y + 10), font, 0.5, (0, 255, 0), thickness)

        # Detect and process objects recognized by the model
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)

                detected_object = class_names[cls].lower()
                camera_width = x2 - x1
                distance = (float(object_dimensions.get(detected_object, 0.15)) * frame_width) / camera_width

                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2

                vector_x = obj_center_x - center_x
                vector_y = obj_center_y - center_y

                angle_deg = (math.degrees(math.atan2(vector_y, vector_x)) + 360) % 360
                direction = f"{int((angle_deg + 30) % 360 / 30) + 1} o'clock"

                cv2.putText(img, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, f"Distance: {distance:.2f} meters", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                if boxes is not None:
                    voice_notification(detected_object, direction, distance)

        cv2.imshow("Webcam", img)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
