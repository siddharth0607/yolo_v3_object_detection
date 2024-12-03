import numpy as np
import cv2

# Load YOLO
yolo = cv2.dnn.readNet(r"E:\yolo model and weights\yolov3.weights", r"E:\yolo model and weights\yolov3.cfg")

# Load class labels
classes = []
with open(r"E:\yolo model and weights\coco.names", 'r') as f:
    classes = f.read().splitlines()

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)
    
    # Get output layer names
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    layer_output = yolo.forward(output_layer_name)

    boxes = []
    confidences = []
    class_labels = []

    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            class_label = np.argmax(scores)
            confidence = scores[class_label]
            if confidence > 0.7:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_labels.append(class_label)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_labels[i]])
            confi = str(round(confidences[i], 2))
            color = colors[class_labels[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confi, (x, y - 10), font, 2, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow('Webcam Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()