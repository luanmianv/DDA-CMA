import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据准备
real_features_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/crossmean/0_features.npy"
fake_features_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/crossmean/4_features.npy"

# 加载特征和标签
real_features = np.load(real_features_path)
fake_features = np.load(fake_features_path)

# 创建标签
real_labels = np.zeros((real_features.shape[0],), dtype=np.int64)  # 标签 0
fake_labels = np.ones((fake_features.shape[0],), dtype=np.int64)  # 标签 1

# 合并数据和标签
features = np.concatenate((real_features, fake_features), axis=0)  # 形状: (2000, 32, 768)
labels = np.concatenate((real_labels, fake_labels), axis=0)        # 形状: (2000,)

# 转换为 PyTorch 张量
features_tensor = torch.tensor(features, dtype=torch.float32)  # 形状: [2000, 32, 768]
labels_tensor = torch.tensor(labels, dtype=torch.int64)       # 形状: [2000]

# 创建数据集
dataset = TensorDataset(features_tensor, labels_tensor)

# 划分数据集 (80% 训练，10% 验证，10% 测试)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建 DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Data split:")
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

# AttentionBlock 和相关类（你的代码）
def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    return nn.GroupNorm(4, channels)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, t, c = x.shape
        x = x.view(b * t, c).unsqueeze(-1)
        x = self.norm(x)
        x = x.squeeze(-1)
        x = x.view(b, t, c)
        
        qkv = self.qkv(x.permute(0, 2, 1))
        h = self.attention(qkv)
        h = self.proj_out(h).permute(0, 2, 1)
        return x + h

# 定义 GRU 模型，嵌入 AttentionBlock
class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.5, bidirectional=True, num_heads=8):
        super(GRUWithAttention, self).__init__()
        self.attention_block = AttentionBlock(channels=input_size, num_heads=num_heads)  # 使用你的跨帧多头注意力模块
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)  # 双向GRU输出

    def forward(self, x):
        # 输入 x 的形状 [batch_size, seq_length, input_size]
        x = self.attention_block(x)  # 应用 AttentionBlock
        _, hidden = self.gru(x)  # GRU 的输出
        # hidden 的形状为 [num_layers * 2, batch_size, hidden_size]
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)  # 拼接最后一层正向和反向隐藏状态
        out = self.fc(out)  # 全连接层
        return out

# 实例化模型
input_size = 768  # 特征维度
hidden_size = 256  # 隐藏层维度
num_classes = 2  # 二分类
num_layers = 2  # GRU 层数

model = GRUWithAttention(
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes,
    num_layers=num_layers,
    dropout=0.5,
    bidirectional=True,
    num_heads=4  # Attention 模块头数
).to(device)
print(model)

# 配置训练参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 余弦退火调度器
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# 训练和验证过程（保持原代码一致）
EPOCHS = 35
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # 训练模式
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录损失和准确率
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # 验证
    # 验证模式
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss / total
    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # 调整学习率
    scheduler.step()
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "crossmean/best_gru_attention_model.pth")
        print("Saved Best Model!")

print("Training complete.")

# 加载 real 和 fake 特征
real_features = np.load("/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/crossmean/0_features.npy")
fake_features = np.load("/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/crossmean/3_features.npy")

# 创建对应的标签
real_labels = np.zeros((real_features.shape[0],), dtype=np.int64)  # 标签 0
fake_labels = np.ones((fake_features.shape[0],), dtype=np.int64)  # 标签 1

# 定义采样比例
real_ratio = 1  
fake_ratio = 1

# 计算采样大小
real_sample_size = int(real_features.shape[0] * real_ratio)
fake_sample_size = int(fake_features.shape[0] * fake_ratio)

# 按顺序采样
real_sampled = real_features[:real_sample_size]  # 从前 real_sample_size 个选择
fake_sampled = fake_features[:fake_sample_size]  # 从前 fake_sample_size 个选择

# 创建对应的标签
real_labels_sampled = np.zeros((real_sampled.shape[0],), dtype=np.int64)  # 标签 0
fake_labels_sampled = np.ones((fake_sampled.shape[0],), dtype=np.int64)  # 标签 1

# 合并采样后的数据和标签
test_features = np.concatenate((real_sampled, fake_sampled), axis=0)  # 合并数据
test_labels = np.concatenate((real_labels_sampled, fake_labels_sampled), axis=0)  # 合并标签

# 转换为 PyTorch 张量
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.int64)

# 创建测试集 DataLoader
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载最佳模型
model.load_state_dict(torch.load("crossmean/best_gru_attention_model.pth"))
model.eval()

from sklearn.metrics import roc_auc_score

# 测试模型
test_loss, correct, total = 0.0, 0, 0
all_labels = []  # 保存真实标签
all_scores = []  # 保存预测为类别1的概率

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 保存真实标签和预测概率
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # 假设类别1为目标类别

# 计算测试损失和准确率
test_loss = test_loss / len(test_loader.dataset)
test_acc = correct / total * 100

# 计算AUC
auc = roc_auc_score(all_labels, all_scores)

# 输出结果
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"AUC: {auc:.4f}")