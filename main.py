import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

# ==== SETTINGS ====
LABEL = "jump"  # <-- เปลี่ยน label ตามท่าที่จะบันทึก
SAVE_EVERY_N_FRAME = 5  # บันทึกทุกกี่เฟรม
OUTPUT_FOLDER = "pose_dataset"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== INITIALIZE MEDIAPIPE ====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# ==== INITIALIZE WEBCAM ====
cap = cv2.VideoCapture(0)
frame_count = 0

# ==== CSV FILE ====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"{OUTPUT_FOLDER}/{LABEL}_{timestamp}.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# เขียน header (label + 33 จุด * (x,y,z,visibility))
headers = ['label']
for i in range(33):
    headers += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
csv_writer.writerow(headers)

print(f"[INFO] Collecting data for label '{LABEL}'... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # วาดจุด pose บน frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # เก็บทุก N frame
        if frame_count % SAVE_EVERY_N_FRAME == 0:
            row = [LABEL]
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            csv_writer.writerow(row)
            print(f"[INFO] Saved frame {frame_count}.")

    frame_count += 1

    cv2.putText(frame, f'Label: {LABEL}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Pose Data Collector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==== CLEAN UP ====
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"[INFO] Data saved to {csv_filename}")
