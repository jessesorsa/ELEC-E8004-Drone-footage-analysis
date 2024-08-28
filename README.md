# ELEC-E8004-Drone-footage-analysis
This project was done as a part of Aalto University course ELEC-E8004 Project Work

## Project
The project included two main parts:
1. Identifying and locating light sources from nighttime drone photos using ML methods
2. Calculating luminance values and mapping them from nighttime drone photos


### Structure
The complete-model folder includes the final project (other folders were used for development)

Run the software by running runner.py
The runner.py imports the luminance and ML models, runs both models, and outputs an image where luminance values and bounding boxes are overlayed with the a test image

### ML model
The predict.py uses the best.pt model. best.pt uses Ultralytics YOLOv8 as the base model, and was fine tuned using a unique set of aerial drone photos taken during the project.
Training data annotations were made manually using CVAT platform

### Luminance
Luminance is calculated from the images with traditional computer vision methods using OpenCV library tools
