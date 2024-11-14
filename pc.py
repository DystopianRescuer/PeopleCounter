#!/bin/python

from ultralytics import YOLO
import cv2
import math
import argparse
import sys
from sort.sort import *


# Some variables to track
drawing = False
start_point = None
end_point = None
counter = 0
tracker = Sort()
crossed_ids = {}


def count_people(capturer, start_point, end_point):
    global counter
    global crossed_ids
    """ Start counting people using the two points provided to make the line """
    startX, startY = start_point
    endX, endY = end_point

    model = YOLO("yolov8n")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
              ]

    while True:
        # take the image
        ret, img= capturer.read()


        # Analyzes the image
        results = model(img, stream=True)

        detections = []
        # For every result...
        for r in results:

            # First draws the line for user reference
            cv2.line(img, start_point, end_point,(0, 255, 0), 2)

            # Draws the counter
            cv2.putText(img, f"Counter: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Takes all boxes provided by the model
            boxes = r.boxes
        
            # Takes its boxes and paints em' all
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # class name
                cls = int(box.cls[0])

                # Adding people objects to detections, and verifying the position.
                if classNames[cls] == "person":
                    detections.append([x1, y1, x2, y2, confidence])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                # Convertir detections a numpy array antes de pasarlo a SORT y verificar dimensiones

            tracked_objects = []
            if len(detections) > 0:
                sortDetections = np.array(detections)
                if sortDetections.shape[1] != 5:
                    sortDetections = np.reshape(sortDetections, (-1, 5))
            else:
                   sortDetections = np.empty((0, 5))

            # Solo actualiza el rastreador si hay detecciones
            if sortDetections.size > 0:
              tracked_objects = tracker.update(sortDetections)
            else:
               tracked_objects = []

            for obj in tracked_objects:
                x1_box, y1_box, x2_box, y2_box, obj_id = map(int, obj)
                cv2.putText(img, f"ID {obj_id}", (x1_box, y1_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Calcular el centro del bounding box
                center_x = int((x1_box + x2_box) / 2)
                center_y = int((y1_box + y2_box) / 2)
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

                # Verificar si el centro cruza la línea
                if startY <= center_y <= endY and (startX <= center_x <= endX):
                     if obj_id not in crossed_ids:
                         counter += 1
                         crossed_ids[obj_id] = True  # Marcar como contado
                         print(f"Persona cruzando la línea! Total: {counter}")

         # Shows the image
        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            break



def line_selector(image):
    """ Function to let user decide which line gon be used """
    global start_point, end_point

    # Window's title
    title = "Draw a line - Press 'q' after that"

    # Show the image and record events 
    cv2.imshow(title, image)
    cv2.setMouseCallback(title, draw_line)

    # Update image while user drawing the line 
    while True:
        temp_image = image.copy()
        
        if start_point and end_point:
            cv2.line(temp_image, start_point, end_point, (0, 255, 0), 2)
        
        # Shows updated image
        cv2.imshow(title, temp_image)
        
        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the window
    cv2.destroyAllWindows()
    
    return start_point, end_point 


def draw_line(event, x, y, flags, param):

    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # Cuando se presiona el botón izquierdo del mouse, comienza el trazo
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Actualiza la posición del punto final mientras el mouse se mueve
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Cuando se suelta el botón izquierdo del mouse, finaliza el trazo
        drawing = False
        end_point = (x, y)


def parse_args():
    """ Reads flags """
    parser = argparse.ArgumentParser(description="Program to count how many people have passed through a line")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-c", "--camera",
        action="store_true",
        help="Use the camera as input"
    )
    group.add_argument(
        "-i", "--input",
        type=str,
        help="Use the path provided as input"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    capturer = cv2.VideoCapture(0 if args.camera else args.input)
    ret, img = capturer.read()
    start_point, end_point = line_selector(img)
    count_people(capturer, start_point, end_point)
