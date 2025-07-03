# 🎥 Anti-Piracy Real-Time Detection System

This project is a real-time surveillance solution designed to detect and track unauthorized recording in cinema halls using **YOLOv8**, **Deep SORT**, and **IR glint detection**. It captures offenders using phones or cameras and logs their activity with images and optional face snapshots.

---

## 🚀 Features

- 📱 **Real-Time Object Detection** using YOLOv8 (detects phones/cameras)
- 🔦 **IR Glint Detection** to spot recording devices (like phone flashes or camera lenses)
- 🔁 **Deep SORT Tracking** with Kalman Filter for stable and smooth object tracking
- 🧠 **Face Detection** for identifying the person recording
- ⏳ **5-Minute Rule** to reduce false alarms (detects continuous suspicious activity)
- 📸 **Snapshot Capture** of the scene and optionally face
- 📁 **CSV Logging** of all detections
- 🎥 *(Optional)* Save full video during suspicious activity
- 📧 *(Optional)* Email notifications with image evidence
- 🔊 *(Optional)* Sound alerts for real-time warning

---

## 🛠️ Requirements

Install the required Python packages:

```bash
pip install ultralytics
pip install deep_sort_realtime
pip install opencv-python
pip install numpy

    
