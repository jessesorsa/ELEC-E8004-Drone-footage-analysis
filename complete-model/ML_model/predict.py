from ultralytics import YOLO
import numpy as np
import os
from PIL import Image, ImageDraw


def overlay_box(image_path, corners, scaling_factor):

    box_color = (255, 0, 0)
    edge_width = 5

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw box
    for box in corners:
        # Get corners
        x_min, y_min, x_max, y_max = box
        shape = [x_min/scaling_factor, y_min/scaling_factor,
                 x_max/scaling_factor, y_max/scaling_factor]

        draw.rectangle(shape, outline=box_color, width=edge_width)

    return image


def predict():

    print("ML model predictions")

    project_path = os.path.dirname(os.path.realpath(__file__))

    # Path to best.pt model
    model_path = project_path + '/best.pt'
    model = YOLO(model_path)

    # Path to images folder
    images_path = project_path + '/images/'

    corners_list = []

    for i, image in enumerate(os.listdir(images_path)):

        # Get individual image path
        image_path = images_path + image

        # Make prediction
        results = model(image_path)

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs

            # result.show()
            # get the centers of boxes
            corners = boxes.xyxy
            corners_list.append(corners)

    return corners_list


def add_bounding_boxes(corners_list, scaling_factor):

    # Get the current working directory
    project_path = os.path.dirname(os.path.realpath(__file__))

    # Construct the relative path to the images folder
    relative_folder_path = "../luminance/results/"

    # Combine the current directory with the relative folder path to get the absolute folder path
    images_path = os.path.join(project_path, relative_folder_path)

    for corners, image in zip(corners_list, os.listdir(images_path)):

        # Get individual image path
        image_path = images_path + image

        modified_image = overlay_box(image_path, corners, scaling_factor)

        # Saving the result
        path = project_path + "/results/" + image
        modified_image.save(path)

        # Adding the metadata from the original image if it exists

        # Opening original and saved result
        mod_image = Image.open(path)
        original_image = Image.open(image_path)

        metadata = original_image.info.get("exif")
        if (metadata):
            mod_image.save(path, exif=metadata)
