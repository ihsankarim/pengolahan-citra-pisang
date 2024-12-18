from preprocessing import BananaPreprocessor
from model import create_cnn_model, train_model
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def main():
    # Inisialisasi preprocessor
    preprocessor = BananaPreprocessor()
    
    # Path ke direktori data
    data_dir = "data/raw"
    
    # Load dan preprocess dataset
    print("Loading and preprocessing dataset...")
    images, labels = preprocessor.prepare_dataset(data_dir)
    
    # Augmentasi data
    print("Augmenting data...")
    augmented_images, augmented_labels = preprocessor.augment_data(images, labels)
    
    # One-hot encode labels
    y_onehot = to_categorical(augmented_labels, num_classes=3)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        augmented_images, y_onehot, test_size=0.2, random_state=42
    )
    
    # Buat dan latih model
    print("Creating and training model...")
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Simpan model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/banana_ripeness_model_v2.keras')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
