from ultralytics import YOLO

model = YOLO('yolov8n.pt') # This loads a pretrained model that is recommended for training

results = model.train(data='C:/Users/Jason/OneDrive/Desktop/weather_dataset', epochs=1, imgsz=64)
# setting echochs to 1 for a easy training to see if every works properly
# data is the path to the images 
# Error 13 on my laptop ask zack to see if same 