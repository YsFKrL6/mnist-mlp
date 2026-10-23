import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Model Mimarisi (main.py de ki model mimarisi ile AYNI olacak)
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 2. Eğitilmiş Modeli Yükleme
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPModel().to(device)
model.load_state_dict(torch.load('mlp_mnist_model.pth', map_location=device))
model.eval()

# 3. MNIST Ön İşleme Ayarları 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_and_predict(image_path):
    # A. Görseli Gri Tonlamalı Oku
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
        
    # B. Renkleri Tersine Çevir (Beyaz kağıda siyah yazıyı -> Siyah arka plan, beyaz yazıya çevirir)
    # OTSU thresholding ile kağıt dokusu vs. temizlenir.
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # C. Modeline Uygun 28x28 Boyutuna Yeniden Boyutlandır
    resized_img = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

    # D. PyTorch formatına çevir
    pil_img = Image.fromarray(resized_img)
    tensor_img = transform(pil_img).unsqueeze(0).to(device) # (1, 1, 28, 28) formatı

    # E. Tahmin 
    with torch.no_grad():
        output = model(tensor_img)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    return predicted_class, confidence

# 4. custom_digits Klasöründeki Tüm Görselleri Test Et
digits_dir = './custom_digits'
if not os.path.exists(digits_dir):
    print(f"'{digits_dir}' klasörü bulunamadı. Lütfen oluşturup içine resimleri ekleyin.")
else:
    image_files = sorted(os.listdir(digits_dir))
    predicted_student_id = ""

    print("\n--- El Yazısı ile Yazdığım Okul Numaramın Tahmin Sonuçları ---")
    for file_name in image_files:
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(digits_dir, file_name)
            pred, conf = preprocess_and_predict(full_path)
            
            if pred is not None:
                predicted_student_id += str(pred)
                print(f"Görsel: {file_name: <10} -> Modelin Tahmini: {pred} (Güven Skoru: %{conf:.2f})")

    print(f"\nModelin Tahmin Ettiği Okul Numarası: {predicted_student_id}\n")
