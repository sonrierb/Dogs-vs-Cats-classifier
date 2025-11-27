import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ----------------------------
# Paths & settings
# ----------------------------
image_dir = "/Users/muskanbansal/Downloads/dogs vs cats/data/train"  # adjust to your path
img_size = (100, 100)
batch_size = 32
model_save_path = "saved_model/model.h5"

# ----------------------------
# Load images
# ----------------------------
image_paths = [os.path.join(image_dir, f"{cls}/{img}") 
               for cls in ["cat", "dog"] 
               for img in os.listdir(os.path.join(image_dir, cls)) 
               if img.endswith(('.jpg', '.jpeg', '.png'))]

random.shuffle(image_paths)

X, y = [], []
for path in image_paths:
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
        label = 0 if "/cat/" in path else 1
        y.append(label)
    except:
        continue

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)

# ----------------------------
# Train/Validation Split
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# CNN Model
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# Train the model
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=batch_size,
    verbose=1
)

# ----------------------------
# Save the model
# ----------------------------
os.makedirs("saved_model", exist_ok=True)
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
