import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# โหลด dataset (ต้องมี combined_dataset.csv อยู่ก่อน)
df = pd.read_csv("combined_dataset.csv")

# แยกข้อมูล
X = df.drop("label", axis=1)
y = df["label"]

# แบ่งชุด Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและเทรนโมเดล
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ทำนายและประเมินผล
y_pred = model.predict(X_test)
print("[RESULT] Classification Report:")
print(classification_report(y_test, y_pred))

# บันทึกโมเดล
joblib.dump(model, "pose_classifier.pkl")
print("[INFO] Model saved to pose_classifier.pkl")
