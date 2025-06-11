import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义 AttentionBlock 和 GRUWithAttention
class AttentionBlock(torch.nn.Module):
    def __init__(self, channels, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = torch.nn.LayerNorm(channels)
        self.qkv = torch.nn.Linear(channels, channels * 3)
        self.attention = torch.nn.MultiheadAttention(channels, num_heads)
        self.proj_out = torch.nn.Linear(channels, channels)

    def forward(self, x):
        b, t, c = x.shape
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        h, _ = self.attention(q, k, v)
        h = h.permute(1, 2, 0, 3).reshape(b, t, c)
        return x + self.proj_out(h)

class GRUWithAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.5, bidirectional=True, num_heads=8):
        super(GRUWithAttention, self).__init__()
        self.attention_block = AttentionBlock(input_size, num_heads=num_heads)
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.fc = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        x = self.attention_block(x)
        _, hidden = self.gru(x)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)  # 拼接双向GRU的隐藏状态
        return self.fc(out)

# 加载特征文件和标签
real_features_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/features/realface_features.npy"
fake_features_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/features/FF++_features.npy"

real_features = np.load(real_features_path)
fake_features = np.load(fake_features_path)
real_labels = np.zeros((real_features.shape[0],), dtype=np.int64)
fake_labels = np.ones((fake_features.shape[0],), dtype=np.int64)

features = np.concatenate((real_features, fake_features), axis=0)
labels = np.concatenate((real_labels, fake_labels), axis=0)

features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.int64)
test_dataset = TensorDataset(features_tensor, labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
input_size = 768
hidden_size = 256
num_classes = 2
num_layers = 2
model_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/best_gru_attention_model.pth"

model = GRUWithAttention(
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes,
    num_layers=num_layers,
    dropout=0.5,
    bidirectional=True,
    num_heads=8
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 测试模型
test_loss, correct, total = 0.0, 0, 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_loss /= len(test_loader.dataset)
test_accuracy = correct / total * 100
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")