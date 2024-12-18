import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple
import numpy as np
import os

def create_cnn_model(input_shape: Tuple[int, int, int] = (224, 224, 8), num_classes: int = 3) -> tf.keras.Model:
    # Gunakan Input layer untuk menghindari warning
    inputs = layers.Input(shape=input_shape)
    
    # Data Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Pisahkan fitur warna dan gambar
    image_features = inputs[:, :, :, :5]  # RGB + 2 color channels
    color_features = inputs[:, :, :, 5:]  # Tambahan fitur warna
    
    # Konvolusi khusus gambar
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(image_features)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Blok Konvolusi 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Blok Konvolusi 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Blok Konvolusi 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Flatten gambar
    x = layers.Flatten()(x)
    
    # Proses fitur warna
    color_features_flat = layers.Flatten()(color_features)
    color_dense = layers.Dense(64, activation='relu')(color_features_flat)
    
    # Gabungkan fitur gambar dan warna
    combined = layers.Concatenate()([x, color_dense])
    
    # Dense layers
    x = layers.Dense(256, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(X_train, y_train, X_val, y_val):
    # Buat model
    model = create_cnn_model(input_shape=(224, 224, 8), num_classes=3)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping dan model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=10, 
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    return model, history