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

            # First draws the line for user reference
            cv2.line(img, x1, y1, x2, y2, (0, 255, 0), 2)

            # Takes all boxes by
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


# Some variables need to track the drawing
drawing = False
start_point = None
end_point = None

def line_selector(image):
    """ Function to let user decide which line gon be used """
    global start_point, end_point

    # Window's title
    title = "Draw a line - Press 'q' after that"

    # Show the image and record events 
    cv2.imshow(title, image)
    cv2.setMouseCallback(title, draw_line)

    # Loop para actualizar la imagen mientras se dibuja la línea
    while True:
        temp_image = image.copy()
        
        if start_point and end_point:
            # Dibujar la línea temporal en la imagen
            cv2.line(temp_image, start_point, end_point, (0, 255, 0), 2)
        
        # Mostrar la imagen actualizada
        cv2.imshow(title, temp_image)
        
        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cerrar la ventana y retornar las coordenadas
    cv2.destroyAllWindows()
    
    return start_point[0], start_point[1], end_point[0], end_point[1]


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
    capturer = cv2.VideoCapture(0 if args.camera else args.input)
    ret, img = capturer.read()
    x1, y1, x2, y2 = line_selector(img)
    count_people(capturer, x1, y1, x2, y2)
