import time 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Veri Setinin Yüklenmesi ve Bölünmesi
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# MLP Model Mimarisi (2 Hidden Layer + ReLU)
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),  
            nn.ReLU(),
            nn.Linear(128, 64),       
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPModel().to(device)

# Kayıp Fonksiyonu ve Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim Döngüsü
epochs = 15
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

start_time = time.time()

print("--- Eğitim Başlıyor (CPU) ---")
for epoch in range(epochs):
    # Training 
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train

    # Validation 
    model.eval()
    val_running_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = val_running_loss / total_val
    epoch_val_acc = correct_val / total_val

    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: %{epoch_train_acc*100:.2f} | "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: %{epoch_val_acc*100:.2f}")

end_time = time.time()
training_time = end_time - start_time

# 6. Test Seti Değerlendirmesi
model.eval()
test_loss, correct_test, total_test = 0.0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_acc = correct_test / total_test
print("\n--- DENEY SONUÇLARI ---")
print(f"Toplam Eğitim Süresi: {training_time:.2f} saniye")
print(f"Son Epoch Train Loss: {history['train_loss'][-1]:.4f}")
print(f"Son Epoch Val Loss  : {history['val_loss'][-1]:.4f}")
print(f"Son Epoch Train Acc : %{history['train_acc'][-1]*100:.2f}")
print(f"Son Epoch Val Acc   : %{history['val_acc'][-1]*100:.2f}")
print(f"Test Accuracy       : %{test_acc*100:.2f}")

# Eğitilmiş Modeli Kaydet
torch.save(model.state_dict(), 'mlp_mnist_model_2.pth')

# 7. Grafiklerin Çizdirilmesi
plt.figure(figsize=(12, 5))

# Loss Grafiği
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), history['train_loss'], label='Training Loss', marker='o')
plt.plot(range(1, epochs+1), history['val_loss'], label='Validation Loss', marker='s')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy Grafiği
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), history['train_acc'], label='Training Accuracy', marker='o')
plt.plot(range(1, epochs+1), history['val_acc'], label='Validation Accuracy', marker='s')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('accuracy_loss_metrics_2.png')
plt.show()