# Real-Time Object Detection

This is a Python script that performs real-time object detection using the YOLOv3 algorithm. It utilizes the OpenCV library and the pre-trained YOLOv3 model to detect objects in a video stream.

## Setup

1. Download the YOLOv3 weights file (`yolov3.weights`) and the configuration file (`yolov3.cfg`) from the official YOLO website.

2. Download the class names file (`coco.names`) from the official YOLO website. This file contains the names of the objects that the YOLO model can detect.

3. Download or capture a video file to be used for object detection. Update the `cap` variable in the script to specify the path to the video file.

## Usage

1. Ensure that you have installed the required dependencies by running the following command:

   ```
   pip install opencv-python
   ```

2. Run the script by executing the following command:

   ```
   python object_detection.py
   ```

3. The script will open a new window showing the real-time object detection on the video. Detected objects will be bounded by rectangles, and their corresponding labels and confidence scores will be displayed.

4. To exit the script, press the 'q' key.

## Dependencies

- OpenCV: A computer vision library used for image and video processing.

## Model and Configuration

The YOLOv3 model is loaded using the `cv2.dnn.readNet()` function, with the weights file and configuration file provided. The script utilizes CUDA acceleration for faster inference by setting the backend and target to CUDA.

## Classes

The class names for the detected objects are loaded from the `coco.names` file. The file contains a list of object names, and each object is assigned a unique class ID.

## Object Detection

The script reads frames from the specified video file and applies the YOLOv3 algorithm for object detection. Detected objects with confidence scores above a threshold (0.5) are extracted, and their bounding boxes are drawn on the frames. The labels and confidence scores are displayed next to the bounding boxes.

## Performance

The script measures the inference time for each frame and displays it in the console. This can be useful for performance analysis and optimization.

Feel free to use and modify this script for your object detection needs. If you have any questions or suggestions, please feel free to reach out. Happy object detection!
