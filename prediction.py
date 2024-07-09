import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

# Load data YAML
with open('yolo5/data.yaml', mode='r') as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)

labels = data_yaml['names']

# Load YOLO model from ONNX file
yolo = cv2.dnn.readNetFromONNX('yolo5/runs/train/Model2/weights/best.onnx')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Read input image 6
img = cv2.imread('dataset/tester/banana-1.jpg')
image = img.copy()

# Get original image dimensions
image_h, image_w = image.shape[:2]

# Prepare input for YOLO
INPUT_WH_YOLO = 640  # Size YOLO expects
blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
yolo.setInput(blob)

# Get predictions from YOLO
preds = yolo.forward()

# Processing predictions
detections = preds[0]
boxes = []
confidences = []
class_ids = []

confidence_threshold = 0.4 # Increased threshold
class_score_threshold = 0.25  # Increased threshold

for i in range(len(detections)):
    row = detections[i]
    confidence = row[4]
    
    if confidence > confidence_threshold:
        class_scores = row[5:]
        class_id = np.argmax(class_scores)
        class_score = class_scores[class_id]
        
        if class_score > class_score_threshold:
            cx, cy, w, h = row[0:4]

            # Calculate bounding box in original image scale
            x_center = int(cx * image_w / INPUT_WH_YOLO)
            y_center = int(cy * image_h / INPUT_WH_YOLO)
            width = int(w * image_w / INPUT_WH_YOLO)
            height = int(h * image_h / INPUT_WH_YOLO)

            # Convert center coordinates to top-left coordinates
            left = int(x_center - width / 2)
            top = int(y_center - height / 2)

            # Append results
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

# Apply Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

# Visualize detections
if len(indices) > 0:
    for i in indices.flatten():
        box = boxes[i]
        left, top, width, height = box
        cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)
        
        # Convert confidence to percentage
        confidence_percentage = confidences[i] * 100
        label = f"{labels[class_ids[i]]}: {confidence_percentage:.2f}%"
        
        # Calculate text size
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

        # Create background rectangle for text for better readability
        text_x = left
        text_y = top - 5
        cv2.rectangle(img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (0, 255, 0), cv2.FILLED)
        
        # Put text on image
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

cv2.imshow("Banana Maturity Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
