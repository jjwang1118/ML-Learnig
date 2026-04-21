# PyTorch 模型建立與使用指南

## 目錄
- [1. 模型建立基礎](#1-模型建立基礎)
- [2. 常用神經網路層](#2-常用神經網路層)
- [3. 優化器 (Optimizer)](#3-優化器-optimizer)
- [4. GPU 加速](#4-gpu-加速)
- [5. 模型預測](#5-模型預測)
- [6. 完整訓練範例](#6-完整訓練範例)

---

## 1. 模型建立基礎

### 1.1 繼承 nn.Module

PyTorch 中所有神經網路模型都繼承自 `nn.Module`：

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        # 呼叫父類別 nn.Module 的 __init__（必須！）
        super(MyModel, self).__init__()
        # 定義網路層
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 定義前向傳播
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 實例化模型
model = MyModel()
```

#### 🔍 `super()` 詳細說明

**為什麼需要 `super(MyModel, self).__init__()`？**

當你的類別繼承自另一個類別時，`super()` 用來呼叫**父類別的方法**：

```python
super(MyModel, self).__init__()
#     ↑          ↑       ↑
#     |          |       └── 呼叫父類別的 __init__ 方法
#     |          └── 當前實例
#     └── 當前類別名稱
```

**`nn.Module.__init__()` 會初始化什麼？**

| 內部屬性 | 用途 |
|---------|------|
| `_parameters` | 儲存所有可訓練參數 (weights, biases) |
| `_modules` | 儲存所有子模組（如 Linear、Conv2d 層）|
| `_buffers` | 儲存非參數張量（如 BatchNorm 的 running_mean）|
| `training` | 標記模型是否處於訓練模式 |

**如果忘記呼叫會怎樣？**

```python
class BadModel(nn.Module):
    def __init__(self):
        # ❌ 忘記呼叫 super().__init__()
        self.layer1 = nn.Linear(10, 64)

model = BadModel()
# AttributeError: cannot assign module before Module.__init__() call
```

**Python 3 簡化寫法：**

```python
# 以下兩種寫法效果相同
super(MyModel, self).__init__()  # 完整寫法
super().__init__()               # Python 3 簡化寫法（推薦）
```

### 1.2 使用 nn.Sequential

適合簡單的順序模型：

```python
# 方法一：直接傳入層
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 方法二：使用 OrderedDict 命名每一層
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 64)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(64, 32)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(32, 1))
]))
```

### 1.3 查看模型結構

```python
# 印出模型架構
print(model)

# 查看模型參數數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'總參數量: {total_params}')
print(f'可訓練參數量: {trainable_params}')
```

---

## 2. 常用神經網路層

### 2.1 全連接層 (Linear Layer)

```python
# nn.Linear(輸入特徵數, 輸出特徵數)
fc = nn.Linear(in_features=128, out_features=64)

# 數學運算: y = xW^T + b
```

### 2.2 卷積層 (Convolutional Layer)

```python
# 2D 卷積層
conv2d = nn.Conv2d(
    in_channels=3,      # 輸入通道數（如 RGB 為 3）
    out_channels=64,    # 輸出通道數（卷積核數量）
    kernel_size=3,      # 卷積核大小
    stride=1,           # 步長
    padding=1           # 填充
)

# 1D 卷積層（常用於序列數據）
conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
```

### 2.3 池化層 (Pooling Layer)

```python
# 最大池化
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 自適應池化（輸出固定大小）
adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
```

### 2.4 正規化層 (Normalization Layer)

```python
# Batch Normalization
bn = nn.BatchNorm2d(num_features=64)  # 2D 用於圖像
bn1d = nn.BatchNorm1d(num_features=64)  # 1D 用於全連接層

# Layer Normalization
ln = nn.LayerNorm(normalized_shape=64)

# Dropout（防止過擬合）
dropout = nn.Dropout(p=0.5)  # 50% 的神經元會被隨機置零
dropout2d = nn.Dropout2d(p=0.5)  # 用於卷積層
```

### 2.5 激活函數

```python
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)
gelu = nn.GELU()  # Transformer 常用
```

### 2.6 循環神經網路層 (RNN)

```python
# LSTM
lstm = nn.LSTM(
    input_size=128,     # 輸入特徵維度
    hidden_size=256,    # 隱藏層維度
    num_layers=2,       # 層數
    batch_first=True,   # 輸入格式 (batch, seq, feature)
    bidirectional=True  # 是否雙向
)

# GRU
gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)

# 基本 RNN
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
```

### 2.7 Embedding 層

```python
# 用於將離散的索引轉換為連續的向量
embedding = nn.Embedding(
    num_embeddings=10000,  # 詞彙表大小
    embedding_dim=256      # 嵌入向量維度
)

# 使用範例
input_ids = torch.LongTensor([1, 2, 3, 4])
embedded = embedding(input_ids)  # shape: (4, 256)
```

---

## 3. 優化器 (Optimizer)

### 3.1 常用優化器

```python
# SGD（隨機梯度下降）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam（最常用）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW（帶權重衰減的 Adam）
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
```

### 3.2 param_groups（參數組）

優化器內部以 `param_groups`（list）管理參數，**允許不同層使用不同超參數**：

```python
# 單一參數組（最常見）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(optimizer.param_groups[0]['lr'])   # 0.001

# 多個參數組（不同層使用不同學習率）
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 0.01},
    {'params': model.layer2.parameters(), 'lr': 0.001},
])
print(optimizer.param_groups[0]['lr'])   # 0.01
print(optimizer.param_groups[1]['lr'])   # 0.001

# 遍歷所有參數組
for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i}: lr={group['lr']}")

# 動態修改學習率
optimizer.param_groups[0]['lr'] = 0.0001
```

**`param_groups[0]` 的結構：**
```python
{
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0,
    'params': [0, 1, 2, ...]   # 參數的 id 列表
}
```

### 3.3 學習率調度器 (Learning Rate Scheduler)

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.1)

# 1. StepLR — 每隔 step_size 個 epoch 乘以 gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 2. MultiStepLR — 在指定的 epoch 衰減
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# 3. ExponentialLR — 每個 epoch 乘以 gamma
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 4. CosineAnnealingLR — 餘弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)

# 5. ReduceLROnPlateau — 當指標不再改善時降低學習率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# 6. CyclicLR — 週期性在兩個邊界之間調整
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=2000)

# 7. OneCycleLR — 單週期策略（適合快速收斂）
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=100, epochs=50)
```

**使用方式：**
```python
for epoch in range(num_epochs):
    train(...)
    scheduler.step()                    # 大多數 scheduler 直接呼叫
    # scheduler.step(val_loss)          # ReduceLROnPlateau 需傳入監控指標

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, LR: {current_lr}')
```

**注意事項：**
- 必須先 `optimizer.step()`，再 `scheduler.step()`
- `ReduceLROnPlateau` 需要傳入監控指標（如 val_loss）
- `ReduceLROnPlateau` 以 `mode='min'`（loss）或 `mode='max'`（accuracy）判斷方向

---

## 4. GPU 加速

### 4.1 檢查 GPU 可用性

```python
# 檢查 CUDA 是否可用
print(torch.cuda.is_available())

# 查看 GPU 數量
print(torch.cuda.device_count())

# 查看當前 GPU 名稱
print(torch.cuda.get_device_name(0))
```

### 4.2 將模型和數據移至 GPU

```python
# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 將模型移至 GPU
model = model.to(device)

# 將數據移至 GPU
data = data.to(device)
target = target.to(device)
```

### 4.3 多 GPU 訓練

```python
# DataParallel（簡單但效率較低）
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# DistributedDataParallel（推薦用於大規模訓練）
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

---

## 5. 模型預測

### 5.1 基本推理流程

推理前必須做兩件事：`model.eval()` 關閉 Dropout/BatchNorm 訓練行為，以及停用梯度計算：

```python
model.eval()

# 方法一：torch.no_grad()
with torch.no_grad():
    output = model(input_data)

# 方法二：torch.inference_mode()（速度更快，推薦純推理場景）
with torch.inference_mode():
    output = model(input_data)
```

### 5.2 分類任務預測

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# 多分類
with torch.inference_mode():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)   # 取最大值的 index
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

# 二分類（搭配 sigmoid）
with torch.inference_mode():
    logits = model(input_data)
    probability = torch.sigmoid(logits)
    predicted = (probability > 0.5).float()
```

### 5.3 批量預測

```python
from torch.utils.data import DataLoader, TensorDataset

test_data = torch.randn(1000, 10)
test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)

model.eval()
all_predictions = []

with torch.inference_mode():
    for (inputs,) in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_predictions.append(torch.argmax(outputs, dim=1).cpu())

all_predictions = torch.cat(all_predictions)
```

---

## 6. 完整訓練範例

### 6.1 圖像分類完整範例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 數據預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# 載入數據
train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                  download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                 download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 定義模型
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 初始化
model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 訓練函數
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

# 測試函數
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total

# 訓練循環
num_epochs = 20
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print('-' * 50)
    
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f'Best Test Accuracy: {best_acc:.2f}%')
```

---

## 附錄：常用技巧

### A. 權重初始化

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)
```

### B. 凍結層（遷移學習）

```python
# 凍結所有層
for param in model.parameters():
    param.requires_grad = False

# 只訓練最後一層
model.classifier[-1] = nn.Linear(128, num_classes)
for param in model.classifier[-1].parameters():
    param.requires_grad = True
```

### C. 梯度裁剪

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### D. 混合精度訓練

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
