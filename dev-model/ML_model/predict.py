from ultralytics import YOLO
import os
from PIL import Image, ImageDraw


def overlay_box(image_path, corners, box_color=(255, 0, 0), edge_width=15):

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw box
    for box in corners:

        # Get corners
        x_min, y_min, x_max, y_max = box
        shape = [x_min, y_min, x_max, y_max]

        draw.rectangle(shape, outline=box_color, width=edge_width)

    return image


project_path = '/Volumes/LaCie/ML/urban-lighting-project/project/complete-model/ML_model/'

# Path to best.pt model
model_path = project_path + 'best.pt'
model = YOLO(model_path)

# Path to images folder
images_path = project_path + 'images/'

for image in os.listdir(images_path):

    # Get individual image path
    image_path = images_path + image

    # Make prediction
    results = model(image_path)

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        # get the centers of boxes
        corners = boxes.xyxy

        overlay = overlay_box(image_path, corners)

        # Saving the result
        path = project_path + "results/" + image
        overlay.save(path)

        # Adding the metadata from the original image

        # Opening original and saved result
        modified_image = Image.open(path)
        original_image = Image.open(image_path)

        metadata = original_image.info.get("exif")
        if (metadata):
            modified_image.save(path, exif=metadata)
