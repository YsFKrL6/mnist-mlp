# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu proje, MNIST veri kümesini kullanarak Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) mimarisi ile el yazısı rakamları sınıflandırmak ve eğitilen modellerin performanslarını karşılaştırmak amacıyla geliştirilmiştir. Ayrıca modeller, kendi el yazımla yazdığım öğrenci numaramın rakam görselleri üzerinde test edilmiştir. Ödev gereksinimi doğrultusunda **CNN kullanılmamıştır.**

## 1. Proje Yapısı

```text
├── data/                         # MNIST veri seti
├── custom_digits/                # Kendi el yazımla hazırladığım rakam görselleri
├── main.py                       # Birinci modelin eğitim ve değerlendirme kodu
├── predict.py                    # Birinci modelle tahmin kodu
├── mlp_mnist_model.pth           # Birinci modelin eğitilmiş ağırlıkları
├── accuracy_loss_metrics.png     # Birinci deneyin eğitim grafikleri
├── experiment2.py                # İkinci modelin eğitim ve değerlendirme kodu
├── predict_exp2.py               # İkinci modelle tahmin kodu
├── mlp_mnist_model_2.pth         # İkinci modelin eğitilmiş ağırlıkları
├── accuracy_loss_metrics_2.png   # İkinci deneyin eğitim grafikleri
├── requirements.txt              # Gerekli kütüphaneler
└── README.md                     # Proje raporu
```

*Not: Bu listede yalnızca projende gerçekten bulunan dosyaları bırak.*

## 2. Veri Seti ve Ön İşleme

Projede, 0 ile 9 arasındaki el yazısı rakamlarından oluşan MNIST veri seti kullanılmıştır. Görüntüler 28 × 28 piksel boyutunda ve gri tonlamalıdır.

Veri seti aşağıdaki şekilde ayrılmıştır:

| Veri kümesi | Görüntü sayısı |
| :--- | ---: |
| Eğitim | 50.000 |
| Doğrulama | 10.000 |
| Test | 10.000 |
| **Toplam** | **70.000** |

Görüntüler tensör biçimine dönüştürülmüş ve normalize edilmiştir. Normalizasyon için aşağıdaki dönüşüm kullanılmıştır:

```python
transforms.Normalize((0.1307,), (0.3081,))
```

Eğitim verileri modelin öğrenmesi, doğrulama verileri eğitim sürecindeki performansın izlenmesi ve test verileri nihai başarının değerlendirilmesi için kullanılmıştır.

## 3. Model Mimarileri ve Hiperparametreler

Her iki modelde de 28 × 28 boyutundaki görüntüler düzleştirilerek 784 elemanlı vektörlere dönüştürülmüştür. Çıkış katmanında 0–9 rakamlarını temsil eden 10 nöron bulunmaktadır.

### Deney 1: Büyük MLP

- Giriş katmanı: 784 nöron
- Birinci gizli katman: 256 nöron
- İkinci gizli katman: 128 nöron
- Aktivasyon fonksiyonu: ReLU
- Çıkış katmanı: 10 nöron

### Deney 2: Küçük MLP

- Giriş katmanı: 784 nöron
- Birinci gizli katman: 128 nöron
- İkinci gizli katman: 64 nöron
- Aktivasyon fonksiyonu: ReLU
- Çıkış katmanı: 10 nöron

Her iki deneyde kullanılan temel hiperparametreler:

| Parametre | Değer |
| :--- | :--- |
| Optimizasyon algoritması | Adam |
| Öğrenme oranı | 0.001 |
| Eğitim batch size | 64 |
| Epoch sayısı | 15 |
| Kayıp fonksiyonu | CrossEntropyLoss |
| Rastgelelik tohumu | 42 |

## 4. Deney Sonuçları

### Deney 1

İlk deneyde gizli katmanlarında sırasıyla 256 ve 128 nöron bulunan MLP modeli eğitilmiştir.

| Metrik | Değer |
| :--- | ---: |
| Eğitim kaybı (Training Loss) | 0.0122 |
| Doğrulama kaybı (Validation Loss) | 0.1018 |
| Eğitim doğruluğu (Training Accuracy) | %99.61 |
| Doğrulama doğruluğu (Validation Accuracy) | %97.99 |
| Test doğruluğu (Test Accuracy) | %98.23 |
| Eğitim süresi | Yaklaşık 128.10 saniye |

### Deney 2

İkinci deneyde gizli katmanlardaki nöron sayıları 128 ve 64 olarak değiştirilmiş, diğer temel eğitim parametreleri aynı tutulmuştur.

| Metrik | Değer |
| :--- | ---: |
| Eğitim kaybı (Training Loss) | 0.0155 |
| Doğrulama kaybı (Validation Loss) | 0.1400 |
| Eğitim doğruluğu (Training Accuracy) | %99.48 |
| Doğrulama doğruluğu (Validation Accuracy) | %97.20 |
| Test doğruluğu (Test Accuracy) | %97.42 |
| Eğitim süresi | Yaklaşık 146.10 saniye |

### Loss ve Accuracy Grafikleri

**Deney 1**

![Birinci deneyin eğitim ve doğrulama grafikleri](accuracy_loss_metrics.png)

**Deney 2**

![İkinci deneyin eğitim ve doğrulama grafikleri](accuracy_loss_metrics_2.png)

## 5. Modellerin Karşılaştırılması

| Metrik | Deney 1 | Deney 2 |
| :--- | ---: | ---: |
| Gizli katman nöronları | 256, 128 | 128, 64 |
| Eğitim doğruluğu | %99.61 | %99.48 |
| Doğrulama doğruluğu | %97.99 | %97.20 |
| Test doğruluğu | %98.23 | %97.42 |
| Eğitim süresi | 128.10 sn | 146.10 sn |

İlk model, test doğruluğunda ikinci modelden 0.81 yüzde puan daha yüksek sonuç elde etmiştir. Ayrıca ölçülen eğitim süresi daha kısadır.

İkinci model daha az nöron kullanmasına rağmen daha uzun sürede eğitilmiştir. Bu durum, model boyutunun tek başına eğitim süresini belirlemediğini gösterir. Bilgisayarın işlem yükü ve ölçüm koşulları da süreyi etkileyebilir.

Bu deneylerde ilk model, test doğruluğu açısından daha başarılı olmuştur.

## 6. Kendi El Yazısı Görsellerimin Değerlendirilmesi

Modeller, kendi el yazımla yazdığım 11 haneli öğrenci numaramın rakam görselleri üzerinde test edilmiştir.

**Gerçek öğrenci numarası:** `24120205064`

Rakamlar ayrı görsellere ayrılmış ve `custom_digits/` klasörüne yerleştirilmiştir. Görseller, tahmin kodunda kullanılan ön işleme adımlarından geçirildikten sonra modellere verilmiştir.

### Deney 1 Tahmin Sonuçları

Modelin oluşturduğu tahmin:

`24724295964`

| Dosya adı | Gerçek rakam | Tahmin | Güven skoru | Sonuç |
| :--- | :---: | :---: | :---: | :--- |
| `01.jpeg` | 2 | 2 | %100.00 | Doğru |
| `02.jpeg` | 4 | 4 | %100.00 | Doğru |
| `03.jpeg` | 1 | 7 | %94.88 | Yanlış |
| `04.jpeg` | 2 | 2 | %90.53 | Doğru |
| `05.jpeg` | 0 | 4 | %94.77 | Yanlış |
| `06.jpeg` | 2 | 2 | %99.99 | Doğru |
| `07.jpeg` | 0 | 9 | %99.86 | Yanlış |
| `08.jpeg` | 5 | 5 | %100.00 | Doğru |
| `09.jpeg` | 0 | 9 | %99.93 | Yanlış |
| `10.jpeg` | 6 | 6 | %100.00 | Doğru |
| `11.jpeg` | 4 | 4 | %100.00 | Doğru |

**Doğru tahmin:** 7/11 — **%63.64**

### Deney 2 Tahmin Sonuçları

Modelin oluşturduğu tahmin:

`24974295964`

| Dosya adı | Gerçek rakam | Tahmin | Güven skoru | Sonuç |
| :--- | :---: | :---: | :---: | :--- |
| `01.jpeg` | 2 | 2 | %99.99 | Doğru |
| `02.jpeg` | 4 | 4 | %100.00 | Doğru |
| `03.jpeg` | 1 | 9 | %99.14 | Yanlış |
| `04.jpeg` | 2 | 7 | %90.52 | Yanlış |
| `05.jpeg` | 0 | 4 | %96.49 | Yanlış |
| `06.jpeg` | 2 | 2 | %99.98 | Doğru |
| `07.jpeg` | 0 | 9 | %99.97 | Yanlış |
| `08.jpeg` | 5 | 5 | %100.00 | Doğru |
| `09.jpeg` | 0 | 9 | %99.92 | Yanlış |
| `10.jpeg` | 6 | 6 | %100.00 | Doğru |
| `11.jpeg` | 4 | 4 | %99.95 | Doğru |

**Doğru tahmin:** 6/11 — **%54.55**

*Not: Güven skoru, modelin seçtiği sınıfa ilişkin tahmin olasılığını gösterir; tahminin doğru olduğunu garanti etmez. Yüzde değerleri yuvarlanmıştır.*

## 7. Genel Değerlendirme ve Tartışma

### MNIST Veri Setindeki Performans

İlk model %98.23, ikinci model ise %97.42 test doğruluğu elde etmiştir. Bu sonuçlar, her iki MLP modelinin de MNIST veri setindeki rakamları başarılı bir şekilde sınıflandırabildiğini göstermektedir.

### Kendi El Yazısı Görsellerindeki Performans

Kendi el yazımla hazırladığım görüntülerde ilk model %63.64, ikinci model ise %54.55 rakam bazlı doğruluk elde etmiştir. Bu oranların MNIST test sonuçlarından düşük olması, eğitim verileri ile kendi görüntülerim arasındaki farklılıklarla ilişkili olabilir.

Olası nedenler şunlardır:

- Rakamların yazım biçimi ve eğimi
- Görüntülerin kırpılması ve rakamın görüntü içindeki konumu
- Rakamların kalınlığı ve boyutu
- Görüntülerin arka planı ve ön işleme adımları
- MNIST görüntüleri ile kendi el yazısı görüntülerim arasındaki dağılım farklılıkları

Bazı rakamların yüksek güven skoruyla yanlış tahmin edilmesi, modelin kendisine verilen görüntüyü belirli bir sınıfa güçlü biçimde benzetebildiğini ancak bu tahminin her zaman doğru olmadığını göstermektedir.

### Aşırı Öğrenme (Overfitting)

İkinci deneyde doğrulama kaybı altıncı epoch'ta en düşük değerine ulaşmıştır. Sonraki epoch'larda eğitim kaybı genel olarak azalmaya devam ederken doğrulama kaybının yükselmesi, aşırı öğrenme belirtisi olabilir. Bununla birlikte, yalnızca bu gözlemle kesin bir sonuca varmak yerine eğitim ve doğrulama eğrilerinin birlikte değerlendirilmesi gerekir.

### Sonuç

Deney sonuçlarına göre ilk model, MNIST test verilerinde ve kendi el yazısı görüntülerimde ikinci modelden daha başarılı olmuştur. Bu çalışma, gizli katman boyutlarının model performansını etkilediğini ve MNIST üzerindeki yüksek başarının gerçek el yazısı görüntülerinde aynı başarıyı garanti etmediğini göstermiştir.

## 8. Projeyi Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

Birinci modeli eğitmek için:

```bash
python main.py
```

Birinci modelle tahmin yapmak için:

```bash
python predict.py
```

İkinci modeli eğitmek için:

```bash
python experiment2.py
```

İkinci modelle tahmin yapmak için:

```bash
python predict_exp2.py
```

Bu komutların çalışabilmesi için ilgili Python dosyalarının proje klasöründe bulunması gerekir.

## 9. Kullanılan Kütüphaneler

- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow

Kütüphanelerin sürümleri `requirements.txt` dosyasında belirtilmiştir.
