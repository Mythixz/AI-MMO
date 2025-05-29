import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv("combined_dataset.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape input for Conv1D: (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Accuracy: {acc:.4f}")

# Save model and label encoder
model.save("pose_cnn_model.h5")
joblib.dump(le, "pose_label_encoder.pkl")
print("[INFO] Model saved to pose_cnn_model.h5")
