import cv2
from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import cvzone  

app = Flask(__name__)

class Detection:
    def __init__(self):
        model_path = os.path.join(os.getcwd(), 'object_detection', 'yolov8n.pt')
        self.model = YOLO(model_path)

    def predict(self, img, classes=[], conf=0.7):  # Increased confidence threshold
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)
        return results

    def predict_and_detect(self, img, conf=0.7, person_class=0, phone_class=67, proximity_threshold=700, rectangle_thickness=6, text_thickness=3):
        results = self.predict(img, conf=conf)

        # Store the coordinates for people and smartphones
        person_boxes = []
        phone_boxes = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                # Check for "person" class and "cell phone" class (COCO: person=0, cell phone=67)
                if class_id == person_class:
                    person_boxes.append(box)
                elif class_id == phone_class:
                    # Apply further filtering after detecting the phone
                    if self.is_valid_smartphone(box.xyxy[0], img):  # Check with additional filters
                        phone_boxes.append(box)

        # Check if any smartphones are near a person's hands
        for person_box in person_boxes:
            person_coords = person_box.xyxy[0]

            # Define the region representing the lower part of the person’s bounding box
            hand_region_top = person_coords[1] + (2/3) * (person_coords[3] - person_coords[1])

            # Draw bounding box for person using cvzone (default: blue)
            img = cvzone.cornerRect(img, (int(person_coords[0]), int(person_coords[1]), int(person_coords[2] - person_coords[0]), int(person_coords[3] - person_coords[1])), colorR=(255, 0, 0), t=rectangle_thickness, l=30)
            cvzone.putTextRect(img, "Person", (int(person_coords[0]), int(person_coords[1]) - 10), scale=1, thickness=text_thickness, colorR=(255, 0, 0))

            # Now look for smartphones within the hand region of the person
            for phone_box in phone_boxes:
                phone_coords = phone_box.xyxy[0]

                # Check if phone's y-coordinates fall within the lower third region of the person's bounding box (assumed hand region)
                if self.is_in_hand_region(person_coords, phone_coords, hand_region_top):
                    # Highlight the person with a red box if the smartphone is within the hand region
                    img = cvzone.cornerRect(img, (int(person_coords[0]), int(person_coords[1]), int(person_coords[2] - person_coords[0]), int(person_coords[3] - person_coords[1])), colorR=(0, 0, 255), t=rectangle_thickness)
                    cvzone.putTextRect(img, "Person with Smartphone", (int(person_coords[0]), int(person_coords[1]) - 10), scale=1, thickness=text_thickness, colorR=(0, 0, 255))

                    # Highlight the smartphone as well
                    img = cvzone.cornerRect(img, (int(phone_coords[0]), int(phone_coords[1]), int(phone_coords[2] - phone_coords[0]), int(phone_coords[3] - phone_coords[1])), colorR=(0, 0, 255), t=rectangle_thickness)
                    cvzone.putTextRect(img, "Smartphone", (int(phone_coords[0]), int(phone_coords[1]) - 10), scale=1, thickness=text_thickness, colorR=(0, 0, 255))

                else:
                    # If not in hand region, draw the smartphone with a purple box (not held by person)
                    img = cvzone.cornerRect(img, (int(phone_coords[0]), int(phone_coords[1]), int(phone_coords[2] - phone_coords[0]), int(phone_coords[3] - phone_coords[1])), colorR=(255, 0, 255), t=rectangle_thickness)
                    cvzone.putTextRect(img, "Smartphone", (int(phone_coords[0]), int(phone_coords[1]) - 10), scale=1, thickness=text_thickness, colorR=(255, 0, 255))
        return img, results

    def is_in_hand_region(self, person_coords, phone_coords, hand_region_top):
        phone_center_y = (phone_coords[1] + phone_coords[3]) / 2
        phone_center_x = (phone_coords[0] + phone_coords[2]) / 2

        # Check if the phone's bounding box is within the hand region
        if phone_center_y >= hand_region_top and person_coords[0] <= phone_center_x <= person_coords[2]:
            return True
        return False

    # Enhanced filtering with Edge and Texture Analysis
    def is_valid_smartphone(self, box, img, min_ratio=1.5, max_ratio=2.5):
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = height / width

        # Apply aspect ratio filter first
        if not (min_ratio <= aspect_ratio <= max_ratio):
            return False

        # Apply edge detection to check for curved corners (common in smartphones)
        roi = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        edges = cv2.Canny(roi, 100, 200)  # Perform edge detection

        # Check if there are sharp corners (e.g., for tablets/lunch boxes)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        corners = len(contours)  # Number of corners
        if corners > 4:
            return False  # Discard if the object has too many sharp edges (not a smartphone)

        # Apply texture filter (optional): Smartphones often have specific textures (screen, buttons)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_val = cv2.mean(gray_roi)[0]
        if mean_val < 50:  # Arbitrary threshold for texture
            return False  # Reject if the texture is too uniform (e.g., lunch boxes)

        return True  # Passed all checks, likely to be a smartphone

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, conf=0.7)
        return result_img


detection = Detection()


@app.route('/')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (720, 576))
        if frame is None:
            break
        frame = detection.detect_from_image(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
    
    #http://localhost:8000 

