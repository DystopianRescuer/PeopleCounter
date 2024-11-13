#!/bin/python

from ultralytics import YOLO
import cv2
import math
import argparse



def count_people(capturer, x1, y1, x2, y2):
    """ Start counting people using the two points provided to make the line """

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
    
        # For every result...
        for r in results:
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
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

         # Shows the image
        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            break


def line_selector(img):
    """ Function to let user decide which line gon be used """
    
    return 0, 0, 0, 0


def parse_args():
    """ Reads flags """
    parser = argparse.ArgumentParser(description="Process camera or video input.")

    # Camera as input flag, no params
    parser.add_argument(
        "-c", "--camera",
        action="store_true",
        default=False,
        help="Usar la cámara en lugar de un archivo de video"
    )

    # Input flag expected to receive a path
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Ruta del archivo de video"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.camera:
        capturer = cv2.VideoCapture(0)
        capturer.set(3, 640)
        capturer.set(4, 480)
        x1, y1, x2, y2 = line_selector(capturer.read())
        count_people(capturer, x1, y1, x2, y2)
    else:
        capturer = cv2.VideoCapture(args.input)
