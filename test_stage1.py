import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集路径
root_dir = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet"
test_dir = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/train"  # 替换为测试数据集路径

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

import os
from torchvision.datasets import ImageFolder

def is_image_file(file_path):
    extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
    return os.path.splitext(file_path)[-1].lower() in extensions

# 加载测试集，添加文件过滤逻辑
test_dataset = ImageFolder(test_dir, transform=data_transforms, is_valid_file=is_image_file)

# 加载测试集
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

# 加载类别名称
class_names = test_dataset.classes
print("Classes:", class_names)

# 加载模型
from models.swin_transformer import SwinTransformer  # 引入 Swin Transformer 模型

net = SwinTransformer(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=10,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=True,
    drop_rate=0.0,
    drop_path_rate=0.2,
    ape=False,
    patch_norm=True,
    use_checkpoint=True
)

# 加载训练好的模型权重
model_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/saved_models/faceswap_cutmix/model_epoch_27.pth"  # 替换为实际模型路径
net.load_state_dict(torch.load(model_path, map_location=device), strict=True)
net = net.to(device)
net.eval()  # 设置为评估模式

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 开始测试
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for data in tqdm(test_loader, "Testing:"):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 计算测试损失和准确率
test_loss /= len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")