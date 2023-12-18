from ultralytics import YOLO
import yaml

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

with open('ocr.yaml') as f:
    setting = yaml.load(f, Loader=yaml.FullLoader)

# Train the model
results = model.train(data='settings/ocr.yaml', 
                      workers=16,
                      batch=64, 
                      epochs=20,
                      patience=10,
                      device=[0, 1])