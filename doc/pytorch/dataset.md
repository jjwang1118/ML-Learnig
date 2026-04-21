# PyTorch Dataset 和 DataLoader

## 目录
1. [Dataset 基础](#dataset-基础)
2. [DataLoader 基础](#dataloader-基础)
3. [自定义 Dataset](#自定义-dataset)
4. [常用数据集](#常用数据集)
5. [数据增强](#数据增强)
6. [实用技巧](#实用技巧)

---

## Dataset 基础

### 什么是 Dataset？
`Dataset` 是 PyTorch 中用于表示数据集的抽象类。它负责存储样本和对应的标签。

### Dataset 的两种类型

#### 1. Map-style Dataset（映射式数据集）
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        # 初始化数据
        self.data = [1, 2, 3, 4, 5]
        self.labels = [0, 1, 0, 1, 0]
    
    def __len__(self):
        # 返回数据集的大小
        return len(self.data)
    
    def __getitem__(self, idx):
        # 根据索引返回一个样本
        return self.data[idx], self.labels[idx]
```

#### 2. Iterable-style Dataset（迭代式数据集）
```python
from torch.utils.data import IterableDataset

class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return iter(range(self.start, self.end))
```

---

## DataLoader 基础

### 什么是 DataLoader？
`DataLoader` 是 PyTorch 中用于加载数据的迭代器。它提供了批处理、打乱、多进程加载等功能。

### 基本使用
```python
from torch.utils.data import DataLoader

# 创建 DataLoader
dataloader = DataLoader(
    dataset=my_dataset,      # 数据集
    batch_size=32,           # 批次大小
    shuffle=True,            # 是否打乱数据
    num_workers=4,           # 多进程加载数据的进程数
    pin_memory=True,         # 将数据固定在内存中（GPU训练时推荐）
    drop_last=False          # 是否丢弃最后不完整的批次
)

# 遍历数据
for batch_data, batch_labels in dataloader:
    print(batch_data.shape, batch_labels.shape)
```

### DataLoader 重要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | Dataset 实例 | 必需 |
| `batch_size` | 每批加载的样本数 | 1 |
| `shuffle` | 是否在每个 epoch 打乱数据 | False |
| `num_workers` | 用于数据加载的子进程数量 | 0 |
| `collate_fn` | 自定义批次数据的整理函数 | None |
| `pin_memory` | 是否将数据加载到固定内存 | False |
| `drop_last` | 是否丢弃最后不完整的批次 | False |
| `sampler` | 自定义采样策略 | None |

### DataLoader Batch 結構範例

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SampleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(10, 3)           # 10 個樣本，每個 3 個特徵
        self.labels = torch.randint(0, 2, (10,))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataloader = DataLoader(SampleDataset(), batch_size=3, shuffle=False)

for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"第 {batch_idx + 1} 個批次:")
    print(f"  batch_data shape:   {batch_data.shape}")    # torch.Size([3, 3])
    print(f"  batch_labels shape: {batch_labels.shape}")  # torch.Size([3])
```

**輸出說明：**
- `batch_data`：形狀為 `(batch_size, features)`，例如 `[3, 3]` 表示 3 個樣本、每個 3 個特徵
- `batch_labels`：形狀為 `(batch_size,)`，例如 `[3]`
- 最後一個批次：若樣本數不能整除 batch_size，最後批次會較小（10 個樣本 ÷ 3 → 最後批只有 1 個）

---

## 自定义 Dataset

### 1. 图像分类 Dataset
```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含所有图像的目录路径
            transform (callable, optional): 可选的转换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 假设文件名包含标签信息，如 "cat_001.jpg"
        label = 0 if 'cat' in self.images[idx] else 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### 2. CSV 数据 Dataset
```python
import pandas as pd
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): CSV 文件的路径
            transform (callable, optional): 可选的数据转换
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # 获取特征和标签
        features = self.data_frame.iloc[idx, :-1].values
        label = self.data_frame.iloc[idx, -1]
        
        # 转换为 Tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
```

### 3. 文本数据 Dataset
```python
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts (list): 文本列表
            labels (list): 标签列表
            tokenizer: 分词器
            max_length (int): 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

---

## 常用数据集

### 1. 使用 TorchVision 数据集
```python
from torchvision import datasets, transforms

# 下载并加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 2. 常用的 TorchVision 数据集
```python
# CIFAR-10
datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# CIFAR-100
datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# ImageNet
datasets.ImageNet(root='./data', split='train', transform=transform)

# Fashion-MNIST
datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# COCO
datasets.CocoDetection(root='./data/coco', annFile='./data/annotations.json')
```

---

## 数据增强

### 使用 transforms 进行数据增强
```python
from torchvision import transforms

# 训练集数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # 随机裁剪并调整大小
    transforms.RandomHorizontalFlip(),      # 随机水平翻转
    transforms.RandomRotation(15),          # 随机旋转
    transforms.ColorJitter(                 # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),                  # 转换为 Tensor
    transforms.Normalize(                   # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 测试集通常只做基本预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 实用技巧

### 1. 自定义 collate_fn
当样本的大小不一致时，需要自定义 `collate_fn` 来处理批次数据。

```python
def custom_collate_fn(batch):
    """
    自定义批次数据整理函数
    """
    # batch 是一个列表，包含 batch_size 个 (data, label) 元组
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]
    
    # 填充序列到相同长度
    data_padded = torch.nn.utils.rnn.pad_sequence(
        data_list, 
        batch_first=True, 
        padding_value=0
    )
    labels = torch.tensor(label_list)
    
    return data_padded, labels

# 使用自定义 collate_fn
dataloader = DataLoader(
    dataset=my_dataset,
    batch_size=32,
    collate_fn=custom_collate_fn
)
```

### 2. 数据集划分
```python
from torch.utils.data import random_split

# 划分训练集和验证集
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(
    dataset, 
    [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### 3. 使用 Subset 选择部分数据
```python
from torch.utils.data import Subset

# 只使用前 1000 个样本
indices = list(range(1000))
subset = Subset(dataset, indices)
subset_loader = DataLoader(subset, batch_size=32)
```

### 4. 自定义采样器
```python
from torch.utils.data import Sampler
import random

class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self):
        # 自定义采样逻辑
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.data_source)

# 使用自定义采样器
dataloader = DataLoader(
    dataset=my_dataset,
    batch_size=32,
    sampler=CustomSampler(my_dataset)
)
```

### 5. 加权随机采样（处理类别不平衡）
```python
from torch.utils.data import WeightedRandomSampler

# 假设有 3 个类别，样本数分别为 [1000, 100, 10]
class_counts = [1000, 100, 10]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

# 为每个样本分配权重
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

dataloader = DataLoader(
    dataset=my_dataset,
    batch_size=32,
    sampler=sampler
)
```

### 6. 多 GPU 训练的分布式采样
```python
from torch.utils.data.distributed import DistributedSampler

# 在分布式训练中使用
sampler = DistributedSampler(
    dataset=train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4
)
```

### 7. 性能优化建议

```python
# 最佳实践配置
dataloader = DataLoader(
    dataset=my_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,              # 根据 CPU 核心数调整
    pin_memory=True,            # GPU 训练时启用
    persistent_workers=True,    # PyTorch 1.7+ 保持 worker 进程存活
    prefetch_factor=2,          # 每个 worker 预加载的批次数
)
```

**性能提示：**
- `num_workers=0`：单进程加载，适合调试
- `num_workers=4-8`：多进程加载，提高效率
- `pin_memory=True`：使用 GPU 时推荐开启
- 批次大小应根据显存和计算能力调整

### 8. 完整训练示例
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 定义 Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 2. 创建数据
X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000,))

# 3. 创建 Dataset 和 DataLoader
train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=2
)

# 4. 定义模型
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_data, batch_labels in train_loader:
        # 前向传播
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
```

---

## 常见问题

### Q1: num_workers 应该设置多少？
**A**: 通常设置为 CPU 核心数的 1-2 倍，但不要超过数据集大小。从 4 开始尝试，根据实际情况调整。

### Q2: 为什么训练速度慢？
**A**: 检查以下几点：
- 增加 `num_workers`
- 启用 `pin_memory`
- 增大 `batch_size`
- 使用 SSD 存储数据
- 在 `__getitem__` 中避免重复的 I/O 操作

### Q3: 如何处理内存不足的问题？
**A**:
- 减小 `batch_size`
- 减少 `num_workers`
- 在 `__getitem__` 中延迟加载数据
- 使用数据生成器而非全部加载到内存

### Q4: shuffle=True 和 sampler 能同时使用吗？
**A**: 不能。当指定 `sampler` 时，`shuffle` 必须为 `False`。

---

## 参考资源

- [PyTorch 官方文档 - Dataset and DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch 官方文档 - torch.utils.data](https://pytorch.org/docs/stable/data.html)
- [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html)
