import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import uuid

# Создание папки для сохранения графиков и результатов
output_dir = "model_comparison_results"
os.makedirs(output_dir, exist_ok=True)

# Устройство для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Гиперпараметры
batch_size = 128
epochs = 20
learning_rate = 0.001

# Функция для сохранения графиков
def save_plot(fig, filename):
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

# Функция для обучения модели
def train_model(model, train_loader, test_loader, criterion, optimizer, model_name, dataset_name):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_times, inference_times = [], []
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Анализ градиентов
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_time = time.time() - start_time
        train_times.append(train_time)
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Тестирование
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        inference_start = time.time()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"{model_name} Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Сохранение кривых обучения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title(f'{model_name} Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title(f'{model_name} Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    save_plot(fig, f"{dataset_name}_{model_name}_learning_curves.png")
    
    # Сохранение confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    save_plot(fig, f"{dataset_name}_{model_name}_confusion_matrix.png")
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'train_acc': train_accuracies[-1],
        'test_acc': test_accuracies[-1],
        'avg_train_time': np.mean(train_times),
        'avg_inference_time': np.mean(inference_times),
        'total_params': total_params,
        'grad_norm': grad_norm
    }

# Определение моделей для MNIST
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64*7*7, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.ReLU()(out)

class ResNetCNN(nn.Module):
    def __init__(self):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64*14*14, 10)
    
    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Определение моделей для CIFAR-10
class DeepFullyConnectedNet(nn.Module):
    def __init__(self):
        super(DeepFullyConnectedNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32*32*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class ResNetCIFAR(nn.Module):
    def __init__(self):
        super(ResNetCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64*16*16, 10)
    
    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class RegularizedResNetCIFAR(nn.Module):
    def __init__(self):
        super(RegularizedResNetCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64*16*16, 10)
    
    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Загрузка данных MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = datasets.MNIST('data', train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST('data', train=False, transform=mnist_transform)
mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size)

# Загрузка данных CIFAR-10
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar_train = datasets.CIFAR10('data', train=True, download=True, transform=cifar_transform)
cifar_test = datasets.CIFAR10('data', train=False, transform=cifar_transform)
cifar_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
cifar_test_loader = DataLoader(cifar_test, batch_size=batch_size)

# Обучение и сравнение моделей для MNIST
mnist_models = [
    ('FullyConnected', FullyConnectedNet().to(device)),
    ('SimpleCNN', SimpleCNN().to(device)),
    ('ResNetCNN', ResNetCNN().to(device))
]
mnist_results = {}

for model_name, model in mnist_models:
    print(f"\nTraining {model_name} on MNIST...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    results = train_model(model, mnist_train_loader, mnist_test_loader, criterion, optimizer, model_name, 'MNIST')
    mnist_results[model==model_name] = results

# Обучение и сравнение моделей для CIFAR-10
cifar_models = [
    ('DeepFullyConnected', DeepFullyConnectedNet().to(device)),
    ('ResNetCIFAR', ResNetCIFAR().to(device)),
    ('RegularizedResNetCIFAR', RegularizedResNetCIFAR().to(device))
]
cifar_results = {}

for model_name, model in cifar_models:
    print(f"\nTraining {model_name} on CIFAR-10...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    results = train_model(model, cifar_train_loader, cifar_test_loader, criterion, optimizer, model_name, 'CIFAR10')
    cifar_results[model_name] = results

# Сохранение результатов анализа
with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
    f.write("MNIST Results:\n")
    for model_name, res in mnist_results.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"Train Loss: {res['train_loss']:.4f}\n")
        f.write(f"Test Loss: {res['test_loss']:.4f}\n")
        f.write(f"Train Accuracy: {res['train_acc']:.2f}%\n")
        f.write(f"Test Accuracy: {res['test_acc']:.2f}%\n")
        f.write(f"Average Training Time per Epoch: {res['avg_train_time']:.2f}s\n")
        f.write(f"Average Inference Time per Epoch: {res['avg_inference_time']:.2f}s\n")
        f.write(f"Total Parameters: {res['total_params']}\n")
        f.write(f"Final Gradient Norm: {res['grad_norm']:.4f}\n")
    
    f.write("\nCIFAR-10 Results:\n")
    for model_name, res in cifar_results.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"Train Loss: {res['train_loss']:.4f}\n")
        f.write(f"Test Loss: {res['test_loss']:.4f}\n")
        f.write(f"Train Accuracy: {res['train_acc']:.2f}%\n")
        f.write(f"Test Accuracy: {res['test_acc']:.2f}%\n")
        f.write(f"Average Training Time per Epoch: {res['avg_train_time']:.2f}s\n")
        f.write(f"Average Inference Time per Epoch: {res['avg_inference_time']:.2f}s\n")
        f.write(f"Total Parameters: {res['total_params']}\n")
        f.write(f"Final Gradient Norm: {res['grad_norm']:.4f}\n")
        f.write(f"Overfitting (Train-Test Acc Diff): {res['train_acc'] - res['test_acc']:.2f}%\n")

print(f"\nResults and plots saved in {output_dir}")