import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import json
import os

# Dataset Path
dataset_path = "C:/Users/akshita bhardwaj/OneDrive/Desktop/plant_disease/plant_disease_dataset"

# Ensure Output Directory Exists
os.makedirs("models", exist_ok=True)

# Data Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load Training & Validation Data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Increased image size for better feature extraction
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model Selection: Choose Between Custom CNN or Pretrained MobileNetV2
USE_MOBILENET = True  # Set to False if you prefer custom CNN

if USE_MOBILENET:
    base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),  # Update to 128x128
    include_top=False,
    weights="imagenet"
)

    base_model.trainable = False  # Freeze base layers
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for Early Stopping, Best Model Saving, and Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("models/plant_disease_best_model.keras", save_best_only=True, monitor='val_loss')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3     ,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Save Final Model
model.save("models/plant_disease_model.keras")


# Save Class Labels
with open("models/class_labels.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Model training completed and saved!")
