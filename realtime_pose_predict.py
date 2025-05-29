# realtime_pose_predict.py
import cv2
import mediapipe as mp
import joblib
import threading
import pandas as pd
import time
import flask_alert_server  # Flask-based alert server

# Load trained model
model = joblib.load("pose_classifier.pkl")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start alert server in background thread
threading.Thread(target=flask_alert_server.run_flask, daemon=True).start()

# Last alert timestamp
last_alert_time = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Can't read camera feed.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])

        if len(row) == 132:
            feature_names = [f'{axis}{i}' for i in range(33) for axis in ['x', 'y', 'z', 'v']]
            df_row = pd.DataFrame([row], columns=feature_names)

            prediction = model.predict(df_row)[0]
            confidence = model.predict_proba(df_row).max()

            # Alert if abnormal behavior detected (with debounce)
            now = time.time()
            if prediction == "fallingdown" and now - last_alert_time > 5 and confidence > 0.8:
                flask_alert_server.q.put("⚠️ ตรวจพบการล้ม กรุณาตรวจสอบ!")
                last_alert_time = now

            cv2.putText(frame, f"{prediction} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        cv2.putText(frame, "No pose detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-Time Pose Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
