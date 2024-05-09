from ultralytics import YOLO
import os
from PIL import Image, ImageDraw


def overlay_boxes_with_dots(image_path, centers, dot_radius=30, dot_color=(255, 0, 0)):

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw dots
    for center in centers:

        # Calculate the center of the box
        center_x, center_y, width, height = center

        # Draw a dot at the center of the box
        dot_bbox = (
            center_x - dot_radius,
            center_y - dot_radius,
            center_x + dot_radius,
            center_y + dot_radius,
        )
        draw.ellipse(dot_bbox, fill=dot_color)

    return image


# Load your own model
model_path = '/Volumes/LaCie/ML/urban-lighting-project/project/object-identification/model/runs/detect/train6/weights/best.pt'
model = YOLO(model_path)

image_path = '/Volumes/LaCie/ML/urban-lighting-project/project/object-identification/model/datasets/dataset-1/test/images/photo_5895740064812614705_y.jpg'

results = model(image_path)


# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs

    # get the centers of boxes
    centers = boxes.xywh

    overlay = overlay_boxes_with_dots(image_path, centers)

    # Saving the result
    filename = os.path.basename(image_path)
    path = "tests/" + filename
    overlay.save(path)

    # Adding the metadata from the original image

    # Opening original and saved result
    modified_image = Image.open(path)
    original_image = Image.open(image_path)

    metadata = original_image.info.get("exif")
    if (metadata):
        modified_image.save(path, exif=metadata)
