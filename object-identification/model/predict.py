from ultralytics import YOLO

# Load your own model
model_path = '/Volumes/LaCie/ML/urban-lighting-project/project/object-identification/model/runs/detect/train2/weights/best.pt'
model = YOLO(model_path)

image = '/Volumes/LaCie/ML/urban-lighting-project/project/object-identification/model/datasets/test/images/DJI_20240415235257_0108_V.JPG'

results = model(image)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
