import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime

# ตั้งค่าท่าที่จะบันทึก (เช่น 'sit', 'stand', 'fall', 'raise_hand')
LABEL = "stand"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

data = []
print(f"[INFO] Collecting data for label: '{LABEL}'... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        row = []
        for landmark in results.pose_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        row.append(LABEL)
        data.append(row)

    # แสดงภาพจากกล้อง
    cv2.putText(frame, f"Label: {LABEL} | Samples: {len(data)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Pose Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# บันทึกเป็น CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{LABEL}_pose_data_{timestamp}.csv"
df = pd.DataFrame(data)
df.to_csv(filename, index=False)

print(f"[SUCCESS] Data saved to {filename}")
