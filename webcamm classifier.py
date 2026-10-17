import cv2
from ultralytics import YOLO

# You can download a pre-trained finger/gesture YOLO model from Roboflow Universe 
# (e.g., 'fingers-counter.pt' or 'yolov8n-hand-gestures.pt')
model = YOLO("yolov8n.pt")  # replace with your trained/downloaded finger model weights

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Run YOLO inference
    results = model(frame, conf=0.5, verbose=False)

    # YOLO automatically draws boxes and labels on the frame
    annotated_frame = results[0].plot()

    # If your model detects individual fingers:
    # finger_count = len(results[0].boxes)

    cv2.imshow("YOLO Finger Counter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
