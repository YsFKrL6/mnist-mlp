# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu proje, MNIST veri kümesini kullanarak Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) mimarisi ile el yazısı rakamları sınıflandırmak ve eğitilen modelin gerçek hayattaki el yazısı görselleri (öğrenci numarası) üzerindeki başarısını değerlendirmek amacıyla geliştirilmiştir. Ödevde istendiği için modelde CNN  **kullanılmamıştır.**

## 1. Proje Yapısı
```text
├── data/                       # İndirilen MNIST veri seti
├── custom_digits/              # Kendi el yazımla yazdığım okul numarası görselleri (01.jpeg - 11.jpeg)
├── main.py                     # Model mimarisi, eğitim, doğrulama ve test kodu
├── predict.py                  # Kendi görsellerimi ön işleyip modele veren tahmin kodu
├── mlp_mnist_model.pth         # Eğitilmiş modelin ağırlıkları (15 Epoch)
├── accuracy_loss_metrics.png   # Eğitim sürecine ait metrik grafikleri
├── requirements.txt            # Gerekli kütüphaneler listesi
└── README.md                   # Proje raporu

## 2. Model Mimarisi ve Hiperparametreler

- **Giriş Katmanı:** 28 × 28 = 784 nöron (Flatten)
- **1. Gizli Katman:** 256 nöron + ReLU aktivasyonu
- **2. Gizli Katman:** 128 nöron + ReLU aktivasyonu
- **Çıkış Katmanı:** 10 nöron (0-9 sınıfları)
- **Kayıp Fonksiyonu:** Cross-Entropy Loss
- **Optimizasyon:** Adam (Learning Rate: 0.001)
- **Batch Size:** 64 (Eğitim) / 1000 (Doğrulama)
- **Epoch Sayısı:** 15
- **Veri Dağılımı:** 50.000 eğitim, 10.000 doğrulama, 10.000 test

## 3. Deney Sonuçları ve Başarım Metrikleri

Eğitim sürecinde ağırlıklar `torch.manual_seed(42)` ile sabitlenmiştir. CPU üzerinde elde edilen nihai metrikler aşağıda verilmiştir.

| Metrik | Değer |
|---|---:|
| Son Training Loss | 0.0122 |
| Son Validation Loss | 0.1018 |
| Son Training Accuracy | %99.61 |
| Son Validation Accuracy | %97.99 |
| Test Accuracy | %98.23 |
| Eğitim Süresi | Yaklaşık 128.10 saniye |

### Loss ve Accuracy Grafikleri

Modelin 15 epoch boyunca eğitim (training) ve doğrulama (validation) setleri üzerindeki başarım eğrileri aşağıdaki görselle gösterilmektedir.

![Eğitim ve doğrulama metrikleri](accuracy_loss_metrics.png)

## 4. Kendi El Yazısı Görsellerinin Değerlendirilmesi

Boş bir A4 kağıdına yazılan 11 haneli okul numarası görselleri ayrı görsellere bölünerek `custom_digits/` klasörüne eklenmiştir. Görseller, OpenCV kullanılarak gri tonlamaya dönüştürme (Grayscale), eşikleme (Thresholding) ve 28 × 28 boyutuna yeniden ölçeklendirme (Resize) işlemlerinden geçirilmiştir. Ardından `predict.py` dosyası ile modele verilmiştir.

### Tahmin Sonuçları

| Dosya Adı | Modelin Tahmini | Güven Skoru |
|---|---:|---:|
| `01.jpeg` | 2 | %100.00 |
| `02.jpeg` | 4 | %100.00 |
| `03.jpeg` | 7 | %94.88 |
| `04.jpeg` | 2 | %90.53 |
| `05.jpeg` | 4 | %94.77 |
| `06.jpeg` | 2 | %99.99 |
| `07.jpeg` | 9 | %99.86 |
| `08.jpeg` | 5 | %100.00 |
| `09.jpeg` | 9 | %99.93 |
| `10.jpeg` | 6 | %100.00 |
| `11.jpeg` | 4 | %100.00 |

## 5. Genel Değerlendirme ve Tartışma

### 5.1. MLP Performansı

Model, standart MNIST test seti üzerinde %98.23 doğruluk oranına ulaşmıştır. Eğitim doğruluğunun (%99.61), doğrulama doğruluğundan (%97.99) yüksek olması, modelin eğitim verilerinde daha başarılı olduğunu göstermektedir. Bu fark, bir miktar ezberleme (overfitting) ihtimaline işaret edebilir; ancak tek başına kesin kanıt değildir.

### 5.2. Gerçek Hayat Verilerinde MLP Sınırlamaları

- MNIST veri setindeki rakamlar genellikle 28 × 28 piksel boyutunda, gri tonlamalı ve belirli bir ön işleme düzenine sahiptir.
- Kendi el yazımızdaki eğim, çizgi kalınlığı, rakamın merkezden kayması ve arka plan farklılıkları modelin tahminlerini etkileyebilir.
- MLP, görüntüyü 784 elemanlı bir piksel dizisi olarak değerlendirir. Bu nedenle piksellerin konumuna duyarlıdır ve küçük konum değişikliklerinden etkilenebilir.
- CNN mimarileri, evrişim katmanları sayesinde yerel şekil ve kenar özelliklerini öğrenebilir. Bu özellikleri sayesinde görüntü sınıflandırma görevlerinde avantaj sağlayabilirler.

### 5.3. Sonuç

Bu projede, CNN kullanılmadan, en az iki gizli katmana sahip bir MLP modeli ile el yazısı rakam sınıflandırması gerçekleştirilmiştir. Eğitim, doğrulama ve test sonuçları modelin MNIST veri setinde yüksek doğruluk elde ettiğini göstermektedir. Kendi el yazısı görsellerindeki sonuçlar ise model başarısının yalnızca test doğruluğuna değil, görüntülerin hazırlanma biçimine ve eğitim verileriyle benzerliğine de bağlı olduğunu göstermektedir.

## 6. Kullanılan Teknolojiler

- **Python**
- **PyTorch:** Model oluşturma ve eğitim
- **Torchvision:** MNIST veri setini yükleme
- **OpenCV:** El yazısı görsellerini ön işleme
- **Matplotlib:** Eğitim ve doğrulama grafiklerini oluşturma

## 7. Projeyi Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

Modeli eğitmek ve test etmek için:

```bash
python main.py
```

Kendi el yazısı görsellerinde tahmin yapmak için:

```bash
python predict.py
```
