import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
import os

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

# Create folders
os.makedirs("detections", exist_ok=True)
csv_file = open("detections/detection_log.csv", "w", newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Detection Type", "Confidence", "Bounding Box"])

# Detection timer
detection_start_time = None
MINIMUM_DURATION_SECONDS = 300  # 5 minutes

print("✅ System running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    current_time = time.time()

    # IR Glint Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ir_glint_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(frame, "IR Glint", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            ir_glint_detected = True
            csv_writer.writerow([timestamp, "IR Glint", "-", f"{x},{y},{w},{h}"])

    # YOLO Object Detection
    results = model(frame)
    phone_detected = False
    alert_triggered = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        if class_name.lower() in ['cell phone', 'camera', 'mobile phone'] and confidence > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            csv_writer.writerow([timestamp, class_name, f"{confidence:.2f}", f"{x1},{y1},{x2},{y2}"])

    # === Combined Alert Logic ===
    if phone_detected and ir_glint_detected:
        if detection_start_time is None:
            detection_start_time = current_time  # start timer
        else:
            duration = current_time - detection_start_time
            if duration >= MINIMUM_DURATION_SECONDS:
                snapshot_name = f"detections/identified_person_{timestamp.replace(':','-')}.jpg"
                cv2.imwrite(snapshot_name, frame)
                print(f"[PHOTO] Person recording over 5 minutes. Photo saved as: {snapshot_name}")
                cv2.putText(frame, "🚨 PERSON IDENTIFIED - PHOTO TAKEN", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                detection_start_time = None  # reset after saving
    else:
        detection_start_time = None  # reset if interruption

    # UI Display
    if phone_detected and ir_glint_detected:
        cv2.putText(frame, "🚨 RECORDING DETECTED!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    elif phone_detected:
        cv2.putText(frame, "📱 Phone Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    elif ir_glint_detected:
        cv2.putText(frame, "🔦 IR Glint Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    else:
        cv2.putText(frame, "✔️ Clear", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    # Show Output
    cv2.imshow("Anti-Piracy Detector", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()
print("📁 All logs and photos saved in 'detections/' folder.")
