# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu proje, MNIST veri kümesini kullanarak Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) mimarisi ile el yazısı rakamları sınıflandırmak ve eğitilen modelin gerçek hayattaki el yazısı görselleri (öğrenci numarası) üzerindeki başarısını değerlendirmek amacıyla geliştirilmiştir. Ödevde istendiği için modelde CNN **kullanılmamıştır.**

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

## 4. Kendi El Yazısı Görsellerimin Değerlendirilmesi

Boş bir A4 kağıdına yazdığım 11 haneli okul numaramın (**24120205064**) görselleri kırılarak `custom_digits/` klasörüne eklenmiş, OpenCV yardımıyla ön işlemeden (Grayscale, Thresholding, 28x28 Resize) geçirilerek `predict.py` vasıtasıyla modele sunulmuştur.

### Tahmin Sonuçları Tablosu:

| Dosya Adı | Gerçek Rakam | Modelin Tahmini | Güven Skoru | Durum |
| :--- | :---: | :---: | :---: | :---: |
| `01.jpeg` | **2** | 2 | %100.00 | Başarılı |
| `02.jpeg` | **4** | 4 | %100.00 | Başarılı |
| `03.jpeg` | **1** | 7 | %94.88 | Başarısız |
| `04.jpeg` | **2** | 2 | %90.53 | Başarılı |
| `05.jpeg` | **0** | 4 | %94.77 | Başarısız |
| `06.jpeg` | **2** | 2 | %99.99 | Başarılı |
| `07.jpeg` | **0** | 9 | %99.86 | Başarısız |
| `08.jpeg` | **5** | 5 | %100.00 | Başarılı |
| `09.jpeg` | **0** | 9 | %99.93 | Başarısız |
| `10.jpeg` | **6** | 6 | %100.00 | Başarılı |
| `11.jpeg` | **4** | 4 | %100.00 | Başarılı |

> **Genel Doğruluk Oranı:** 7 / 11 (%63.63)

## 5. Genel Değerlendirme ve Tartışma

1. **MLP Performansı:** Eğitim doğruluğunun (%99.61), doğrulama doğruluğundan (%97.99) yüksek olması, modelin eğitim verilerinde daha başarılı olduğunu göstermektedir. Aradaki fark, bir miktar ezberleme (overfitting) ihtimaline işaret edebilir; ancak kesin bir sonuca varmak için eğitim sürecinin tamamı ve kayıp eğrileri birlikte değerlendirilmelidir.
2. **Gerçek Hayat Verilerinde MLP Sınırlamaları:** 
   - Kendi el yazımızla hazırladığımız veri setinde '1' ve '0' rakamlarında yaşanan tahmin sapmalarının temel sebebi, MLP mimarisinin uzamsal değişmezliğe (spatial invariance) sahip olmamasıdır.
   - MNIST verileri piksel olarak tam merkeze oturtulmuş ve standart çizgi kalınlıklarına sahiptir. Kendi el yazımızdaki kişisel stil farklılıkları (örneğin '1' rakamındaki çentik veya '0' rakamındaki hafif eğim) ve piksellerin merkezden ufak kaymaları, piksel tabanlı çalışan MLP'nin farklı özellikler algılamasına yol açabilmektedir.
   - Bu durum, görüntü işleme problemlerinde piksellerin konumundan ziyade kenar, köşe ve şekil özelliklerini öğrenebilen **Evrişimli Sinir Ağlarının (CNN)** olası avantajlarını göstermektedir
