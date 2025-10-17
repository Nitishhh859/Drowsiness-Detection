#  1. Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#2. Load Dataset
data_dir = "/content/train"
labels = {'Open_Eyes': 0, 'Closed_Eyes': 1}
X = []
y = []

for label in labels:
    path = os.path.join(data_dir, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (24, 24))
        X.append(img)
        y.append(labels[label])

X = np.array(X).reshape(-1, 24, 24, 1) / 255.0
y = to_categorical(y, num_classes=2)

print(f"Loaded images: {X.shape}")

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[early_stop], batch_size=64)

#  6. Plot
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.title("Accuracy vs Epochs")
plt.show()

#  7. Test on Video (no webcam)
video_path = "/content/tired-man-driving-on-highway-eyes-closes-drop-and-falls-asleep-behind-wheel-lo-SBV-348554928-preview (1) (1).mp4"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 4 != 0:  # only every 4th frame
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_face = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_face, 1.1, 4)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_face[ey:ey+eh, ex:ex+ew]
            eye = cv2.resize(eye, (24, 24)).reshape(1, 24, 24, 1) / 255.0
            prediction = model.predict(eye)
            status = np.argmax(prediction)

            if status == 1:
                cv2.putText(frame, 'Closed Eyes - Drowsy 😴', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Open Eyes - Alert 🚗', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2_imshow(cv2.resize(frame, (500, 300)))

cap.release()
cv2.destroyAllWindows()
