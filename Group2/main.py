from ultralytics import YOLO
import cv2 

model = YOLO('yolov8n-cls.pt') # This loads a pretrained model that is recommended for training

results = model.train(data=r'C:\Users\User\Documents\GitHub\aiRoadSafety.github.io\Group2\weather_training_set', epochs=1, imgsz=64)

# setting echochs to 1 for a easy training to see if everything works properly
# data is the path to the images 



# object recognition using a video 
# load yolov8 model
model = YOLO('yolov8n.pt')

# load video 
video_path = 'C://Users//Jason//OneDrive//Desktop//test.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
# read frames 
# Using while loop to make sure we break out when the video ends 
while ret:
    ret, frame = cap.read()
    
    
# detect objects and track the objects 
    results = model.track(frame, persist=True)
# by setting persist to true will allow yolov8 to remember the objects it has seen through each frame 
    
# plot results    
    cv2. rectangle 
    cv2. putText
    frame_ = results[0].plot()

# visualize     
    cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break