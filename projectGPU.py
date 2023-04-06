#Developed by Mihir Madkaikar BE EXTC A
import cv2
import numpy as np
import time

# Load YOLOv3 weights and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set input and output layers
layer_names = net.getLayerNames()
if len(layer_names) > 0:
    # Convert numpy array to list
    unconnected_layers = net.getUnconnectedOutLayers().tolist()
    # Update output_layers to remove check for len(i) > 0
    output_layers = [layer_names[i - 1] for i in unconnected_layers]
else:
    print("Error: No layers found in the network.")

# Load video
cap = cv2.VideoCapture("video.avi")

# Process video frames
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    start_time = time.time()
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    end_time = time.time()

    # Extract bounding boxes and confidence scores
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y - 5), font, 1, color, 1)

    # Display output frame
    cv2.imshow("Object Detection", frame)
    print("Inference time: ", end_time - start_time)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
