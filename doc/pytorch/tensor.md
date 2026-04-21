# PyTorch Tensor 操作指南

## 檢查 CUDA 是否可用
```python
torch.cuda.is_available()  # 回傳 True 或 False
```

## 創建 Tensor

### 直接從 List 創建
```python
list=[[],[]]
torch.tensor(data)  # 直接把整個 list 搬入 tensor
```

### 其他創建方法
- `torch.ones(size)` - 創建一個全為 1 的 tensor
- `torch.zeros(size)` - 創建一個全為 0 的 tensor
- `torch.rand(size)` - 創建一個隨機值的 tensor，size=(row, column)
- `torch.arange(start, end, step)` - 創建一個等差數列的 tensor
- `torch.linspace(start, end, steps)` - 創建一個等差數列的 tensor，steps: 總共有幾個數
- `torch.eye(n)` - 創建一個單位矩陣，n: 矩陣大小

## 查看維度
```python
tensor.shape           # 查看完整維度
tensor.shape[0]        # 取得 row 數
tensor.shape[1]        # 取得 column 數
```

## 設定資料型態 (dtype)
```python
torch.tensor(data, dtype=torch.float32)  # 創建時設定為 float32
tensor.type(torch.float32)               # 轉換為 float32
tensor.dtype                             # 查看型態
```

## 取得特定值
```python
tensor[row][column].item()  # 使用 item() 會回傳純量
tensor[row][column]         # 不使用 item() 會回傳一個 tensor
```

## Tensor 計算

### 基本運算
- 兩個 tensor 做運算 `+`, `-`, `*`, `/`, `>`, `<`, `==` 都是對應位置做運算
- **注意**: tensor 的計算要注意維度是否相符

### 加總 (sum)
```python
torch.sum(tensor)              # 回傳整個 tensor 的總和
torch.sum(tensor, dim=0)       # 對 column 做加總，回傳 [column]
torch.sum(tensor, dim=1)       # 對 row 做加總，回傳 [row]
torch.sum(判斷條件的tensor)     # 回傳符合條件的值的總和
```

### 矩陣相乘
```python
torch.matmul(tensor_a, tensor_b)  # 矩陣相乘
```

### 平均值 (mean)
```python
torch.mean(tensor)             # 回傳整個 tensor 的平均值
torch.mean(tensor, dim=0)      # 對 column 做平均，回傳 [column]
torch.mean(tensor, dim=1)      # 對 row 做平均，回傳 [row]
```
- **注意**: Tensor 型態會變成 float32
- 計算相等比例: `torch.mean((tensor1 & tensor2 布林運算).to(torch.float32))`

### 排序與索引
```python
torch.argsort(tensor)          # 排序完後，回傳數據原始 index 位置
torch.argmin(tensor)           # 回傳最小值的 index
torch.argmax(tensor)           # 回傳最大值的 index
```

#### 使用 dim 參數
- 沒有 dim: 回傳整個 tensor 的結果
- 1 維度 tensor: 直接處理該 tensor
- 2 維度 tensor 加入 dim: 對每一 row 或 column 做操作，回傳排序後的 index 位置

## Tensor 重塑

### 改變形狀
```python
tensor.reshape(new_shape)      # 改變 tensor 的形狀，new_shape=(row, column)
tensor.view(new_shape)         # 改變 tensor 的形狀，new_shape=(row, column)
```
- **注意**: row 或 column 可以用 `-1` 表示自動計算

### 攤平
```python
tensor.flatten()               # 把多維度 tensor 攤平成 1 維度
```
