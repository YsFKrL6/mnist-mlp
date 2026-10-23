
# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu proje, MNIST veri kümesini kullanarak Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) mimarisi ile el yazısı rakamları sınıflandırmak ve eğitilen modelin gerçek hayattaki el yazısı görselleri (öğrenci numarası) üzerindeki başarısını değerlendirmek amacıyla geliştirilmiştir. Ödev isterlerine uygun olarak modelde CNN (Evrişimli Sinir Ağı) **kullanılmamıştır.**

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
```

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

Eğitim sürecinde ağırlıklar `torch.manual_seed(42)` ile sabitlenmiştir. CPU üzerinde elde edilen nihai metrikler aşağıdadır:

| Metrik | Değer |
| :--- | :--- |
| **Training Loss** | 0.0122 |
| **Validation Loss** | 0.1018 |
| **Training Accuracy** | %99.61 |
| **Validation Accuracy** | %97.99 |
| **Test Accuracy** | %98.23 |
| **Eğitim Süresi** | ~128.10 saniye |

### Loss ve Accuracy Grafikleri

![Training and Validation Metrics](accuracy_loss_metrics.png)

## 4. Kendi El Yazısı Görsellerinin Değerlendirilmesi

Boş bir A4 kağıdına yazılan 11 haneli okul numarası görselleri kırılarak `custom_digits/` klasörüne eklenmiş, OpenCV yardımıyla ön işlemeden (Grayscale, Thresholding, 28x28 Resize) geçirilerek `predict.py` vasıtasıyla modele sunulmuştur.

### Tahmin Sonuçları Tablosu:

| Dosya Adı | Modelin Tahmini | Güven Skoru |
| :--- | :---: | :---: |
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

1. **MLP Performansı:** Model, standart MNIST test seti üzerinde **%98.23** gibi yüksek bir başarı oranına ulaşmıştır. Eğitim doğruluğunun (%99.61) doğrulama doğruluğundan (%97.99) çok az yüksek olması, modelin ezber yapmadan genel kalıpları öğrenebildiğini göstermektedir.
2. **Gerçek Hayat Verilerinde MLP Sınırlamaları:** 
   - Kendi el yazımızla hazırladığımız veri setinde bazı rakamlarda (örneğin '1' ve '0') yaşanan tahmin sapmalarının temel sebebi, MLP mimarisinin uzamsal değişmezliğe (spatial invariance) sahip olmamasıdır.
   - MNIST verileri piksel olarak tam merkeze oturtulmuş ve standart çizgi kalınlıklarına sahiptir. Kendi el yazımızdaki kişisel stil farklılıkları (çentikler, eğimler) ve piksellerin merkezden ufak kaymaları, piksel tabanlı çalışan MLP'nin farklı özellikler algılamasına yol açabilmektedir.
   - Bu durum, görüntü işleme problemlerinde piksellerin konumundan ziyade kenar, köşe ve şekil özelliklerini öğrenebilen **Evrişimli Sinir Ağlarına (CNN)** neden ihtiyaç duyulduğunu deneysel olarak kanıtlamaktadır.
