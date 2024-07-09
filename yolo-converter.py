import json
import os
import cv2

def convert_labelme_to_yolo(json_dir, image_dir, output_dir, class_mapping):
    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_path = os.path.join(image_dir, data['imagePath'])
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            output_path = os.path.join(output_dir, os.path.splitext(json_file)[0] + '.txt')
            with open(output_path, 'w') as f:
                for shape in data['shapes']:
                    label = shape['label']
                    points = shape['points']

                    x_min = min([point[0] for point in points])
                    y_min = min([point[1] for point in points])
                    x_max = max([point[0] for point in points])
                    y_max = max([point[1] for point in points])

                    x_center = (x_min + x_max) / 2.0 / width
                    y_center = (y_min + y_max) / 2.0 / height
                    bbox_width = (x_max - x_min) / width
                    bbox_height = (y_max - y_min) / height

                    class_id = class_mapping[label]

                    f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

json_dir = 'dataset/jsons/val'
image_dir = 'dataset/images/val'
output_dir = 'dataset/labels/val'
class_mapping = {'mentah': 0, 'matang': 1, 'matang sepenuhnya': 2}

convert_labelme_to_yolo(json_dir, image_dir, output_dir, class_mapping)
