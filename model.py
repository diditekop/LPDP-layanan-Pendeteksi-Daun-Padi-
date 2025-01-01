import numpy as np 
import warnings
warnings.filterwarnings("ignore")

import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

IMAGE_SIZE = (224, 224)
DATA_DIR = "D:\QARIR ACADEMY\(LPDP) Layanan Pendeteksi Daun Padi\Dataset"

X = []
y = []

for category in os.listdir(DATA_DIR):
    category_dir = os.path.join(DATA_DIR, category)
   
    for image_file in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_file)
        
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype('float32') / 255.0  
       
        X.append(img)
        
        if category == "tungro":
            y.append(0)
        elif category == "blast":
            y.append(1)
        elif category == "blight":
            y.append(2)


X = np.array(X)
y = np.array(y)


print("X shape:", X.shape)
print("y shape:", y.shape)

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: tungro, blast, blight
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)


model.save('model_penyakit_padi.h5')