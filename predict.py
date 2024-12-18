from preprocessing import BananaPreprocessor
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import sys

class BananaRipenessPredictor:
    def __init__(self, model_path='models/banana_ripeness_model_v2.keras'):
        self.preprocessor = BananaPreprocessor()
        self.model = load_model(model_path)
        self.label_names = ['Belum Matang', 'Matang', 'Kematangan']

    def predict(self, image_path):
        # Muat dan preprocess gambar
        img = self.preprocessor.load_and_preprocess_image(image_path)
        
        # Preprocess untuk model
        img_processed = self.preprocessor.preprocess_for_model(img)
        
        # Prediksi
        prediction = self.model.predict(np.expand_dims(img_processed, axis=0))[0]
        
        # Cetak hasil prediksi
        print("\nHasil Prediksi Kematangan Pisang:")
        for name, prob in zip(self.label_names, prediction):
            print(f"{name}: {prob*100:.2f}%")
        
        # Tentukan kategori dengan probabilitas tertinggi
        max_index = np.argmax(prediction)
        print(f"\nKesimpulan: Pisang dalam kondisi {self.label_names[max_index]}")

def main():
    if len(sys.argv) < 2:
        print("Gunakan: python predict.py <path_gambar>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predictor = BananaRipenessPredictor()
    predictor.predict(image_path)

if __name__ == "__main__":
    main()