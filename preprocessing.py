import os
import cv2
import numpy as np
from typing import Tuple, List
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import scipy.stats

class BananaPreprocessor:
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        self.input_size = input_size
        self.data_augmentation = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

    def remove_background(self, image):
        # Konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Threshold adaptif
        thresh = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Operasi morfologi untuk membersihkan noise
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Buat mask
        mask = cv2.dilate(opening, kernel, iterations=3)
        
        # Hapus background
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result

    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        # Baca gambar
        img = cv2.imread(image_path)
        
        # Pastikan gambar berhasil dibaca
        if img is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        # Konversi ke RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize gambar
        img = cv2.resize(img, self.input_size)
        
        # Pastikan gambar 3 channel dan tipe data uint8
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        
        # Konversi ke uint8 jika belum
        img = img.astype(np.uint8)
        
        return img

    def preprocess_for_model(self, img: np.ndarray) -> np.ndarray:
        # Pastikan gambar dalam format uint8 dan 3 channel
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        
        # Normalisasi warna dasar
        img_normalized = img.astype('float32') / 255.0
        
        # Color-based preprocessing
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Ekstraksi fitur warna tambahan
        hue_channel = hsv[:,:,0] / 179.0  # Normalisasi hue
        saturation_channel = hsv[:,:,1] / 255.0
        
        # Ekstraksi fitur warna lanjutan
        color_features = self.advanced_color_features(img)
        
        # Pilih fitur paling relevan untuk membedakan kematangan
        # Urutan: hue_mean, hue_std, sat_mean, dominant_hue, green_percentage
        selected_features = color_features[[0, 1, 3, 7, 8]]
        
        # Tile color features to match image dimensions
        # Buat 3 channel terakhir dari fitur warna
        color_features_tiled = np.stack([
            np.full((224, 224), selected_features[0], dtype=np.float32),  # hue_mean
            np.full((224, 224), selected_features[1], dtype=np.float32),  # hue_std
            np.full((224, 224), selected_features[4], dtype=np.float32)   # green_percentage
        ], axis=-1)
        
        # Gabungkan fitur warna - pastikan tepat 8 channel
        img_with_features = np.concatenate([
            img_normalized,  # 3 channel RGB
            hue_channel.reshape(224, 224, 1),  # 4. hue channel
            saturation_channel.reshape(224, 224, 1),  # 5. saturation channel
            color_features_tiled  # Last 3 color features
        ], axis=-1)
        
        return img_with_features

    def prepare_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Scanning directory: {data_dir}")
        
        # Daftar label berdasarkan subdirektori
        labels_map = {
            'belum_matang': 0,
            'matang': 1,
            'kematangan': 2
        }
        
        images = []
        labels = []
        
        # Iterasi melalui setiap kategori
        for category, label_id in labels_map.items():
            category_path = os.path.join(data_dir, category)
            
            # Pastikan direktori kategori ada
            if not os.path.exists(category_path):
                print(f"Warning: Directory {category_path} not found!")
                continue
            
            # Proses setiap gambar dalam kategori
            for img_filename in os.listdir(category_path):
                if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(category_path, img_filename)
                    
                    try:
                        # Muat dan preprocess gambar
                        preprocessed_img = self.load_and_preprocess_image(img_path)
                        
                        # Debug print
                        print(f"Loaded image {img_path}: shape={preprocessed_img.shape}, dtype={preprocessed_img.dtype}")
                        
                        images.append(preprocessed_img)
                        labels.append(label_id)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Konversi ke numpy array
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Total images loaded: {len(images)}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return images, labels

    def augment_data(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            # Augmentasi gambar asli
            original_processed = self.preprocess_for_model(img)
            augmented_images.append(original_processed)
            augmented_labels.append(label)
            
            # Augmentasi gambar yang dimodifikasi
            augmented_img = self.data_augmentation.random_transform(img)
            augmented_processed = self.preprocess_for_model(augmented_img)
            augmented_images.append(augmented_processed)
            augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def advanced_color_features(self, img):
        # Konversi ke HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Ekstraksi fitur warna yang lebih kompleks
        hue = hsv[:,:,0]
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # Hitung statistik warna yang lebih detail
        hue_mean = np.mean(hue) / 179.0
        hue_std = np.std(hue) / 179.0
        hue_skew = scipy.stats.skew(hue.flatten()) / 10.0  # Tambahkan skewness
        
        sat_mean = np.mean(saturation) / 255.0
        sat_std = np.std(saturation) / 255.0
        
        val_mean = np.mean(value) / 255.0
        val_std = np.std(value) / 255.0
        
        # Deteksi warna dominan dengan histogram
        unique, counts = np.unique(hue, return_counts=True)
        dominant_hue = unique[np.argmax(counts)] / 179.0
        
        # Tambahkan fitur tambahan untuk membedakan tahap kematangan
        green_mask = np.logical_and(hue >= 30, hue <= 90)  # Rentang warna hijau
        green_percentage = np.mean(green_mask)
        
        # Gabungkan fitur
        color_features = np.array([
            hue_mean,     # Rata-rata hue
            hue_std,      # Variasi hue
            hue_skew,     # Skewness hue
            sat_mean,     # Rata-rata saturasi
            sat_std,      # Variasi saturasi
            val_mean,     # Rata-rata value
            val_std,      # Variasi value
            dominant_hue, # Hue dominan
            green_percentage  # Persentase area hijau
        ])
        
        return color_features

def main() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Inisialisasi preprocessor
    preprocessor = BananaPreprocessor()
    
    # Path ke direktori data
    data_dir = "data/raw"
    
    # Load dan preprocess dataset
    images, labels = preprocessor.prepare_dataset(data_dir)
    
    # Augmentasi data
    augmented_images, augmented_labels = preprocessor.augment_data(images, labels)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        augmented_images, augmented_labels, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()