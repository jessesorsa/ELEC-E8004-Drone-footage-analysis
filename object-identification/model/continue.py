from ultralytics import YOLO

# Path to previous model
model_path = "/Volumes/LaCie/ML/urban-lighting-project/project/object-identification/model/runs/detect/train6/weights/last.pt"

# Load a model
model = YOLO(model_path)

model.train(resume=True)
