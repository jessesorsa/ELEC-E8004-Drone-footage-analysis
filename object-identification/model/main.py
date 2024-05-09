from ultralytics import YOLO

# Path to previous model
model_path = "/Volumes/LaCie/ML/urban-lighting-project/project/object-identification/model/runs/detect/train2/weights/light-detection-1.pt"

# Load a model
model = YOLO(model_path)

# Use the model
model.train(data="config.yaml", epochs=100)  # train the model
