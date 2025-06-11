import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms
from models.swin_transformer import SwinTransformer
from tqdm import tqdm
from time import time
import random
from torch.utils.data import random_split, DataLoader

from augmentation.mixup import generate_attention_guided_mixup_with_labels
from augmentation.cutmix import DynamicCutMix

# 设置随机种子和设备
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE = 128
EPOCHS = 30
NUM_WORKERS = 8
lr = 5e-5

# 数据集路径
root_dir = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet"
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 加载原始数据集
full_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train1'), data_transforms['train'])

# 按照 8:1:1 比例划分数据集
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 创建 DataLoader
data_loader = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
    'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
}

# 数据集大小
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset),
}
print("Dataset sizes:", dataset_sizes)

# 获取类别名称
class_names = full_dataset.classes
print("Classes:", class_names)

# 定义 Swin Transformer 模型
net = SwinTransformer(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
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
net.load_state_dict(torch.load('swin_tiny_patch4_window7_224.pth')['model'], strict=True)
net.head = nn.Linear(net.head.in_features, 10)
net = net.to(device)

# 初始化 CutMix
target_layer = net.layers[-1].blocks[-1].norm2  # Swin Transformer 的目标层
cutmix = DynamicCutMix(model=net, target_layer=target_layer, device=device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# 创建模型保存目录
save_dir = "saved_models/faceswap_cutmix_333"
os.makedirs(save_dir, exist_ok=True)
from collections import Counter

# 检查每个子集的标签分布
train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
val_labels = [full_dataset.targets[i] for i in val_dataset.indices]
test_labels = [full_dataset.targets[i] for i in test_dataset.indices]

print("Training set label distribution:", Counter(train_labels))  # 输出 {0: 数量, 1: 数量}
print("Validation set label distribution:", Counter(val_labels))
print("Testing set label distribution:", Counter(test_labels))
# 训练日志和损失记录
training_logs = []
train_losses = []


# 加载检查点
start_epoch = 0
checkpoint_path = os.path.join(save_dir, "checkpoint_epoch_latest.pth")
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")

for epoch in tqdm(range(start_epoch, EPOCHS), "Epoch: "):
    net.train()
    optimizer.zero_grad()

    start = time()
    running_loss = 0.0

    tqdm_train = tqdm(enumerate(data_loader['train'], 0), "Train step: ", total=len(data_loader['train']))
    for i, data in tqdm_train:
        inputs, labels = data[0].to(device), data[1].to(device)
        # CutMix 数据增强（20% 概率）
        if epoch >= 11 and epoch <= 20 and np.random.rand() < 0.3:
            cutmix_idx = torch.randperm(inputs.size(0)).to(device)  # 随机索引
            cutmix_inputs = inputs[cutmix_idx]
            cutmix_labels = labels[cutmix_idx]

            # 动态生成 CutMix 数据
            cutmix_inputs_augmented = torch.stack([
                cutmix.augment(inputs[i], cutmix_inputs[i], target_class=labels[i].item()).to(device)
                for i in range(len(inputs))
            ])

            # 替换数据为增强结果
            inputs = cutmix_inputs_augmented

        # 决定是否对当前批次应用 MixUp 数据增强
        elif epoch >= 11 and epoch <= 20 and np.random.rand() < 0.3:  # 20% 概率应用 MixUp
            # 随机生成索引并匹配标签
            mixup_idx = torch.randperm(inputs.size(0)).to(device)
            matched_mask = (labels == labels[mixup_idx])  # 匹配相同标签的样本
            if matched_mask.sum() > 1:  # 至少有一个匹配的样本对
                matched_idx = matched_mask.nonzero(as_tuple=True)[0]

                # 提取匹配的样本对
                inputs_selected = inputs[matched_idx]
                labels_selected = labels[matched_idx]
                mixup_idx_selected = mixup_idx[matched_idx]

                # 调用 Grad-CAM 生成 Attention Map
                def generate_attention_map(input_tensor):
                    net.eval()
                    input_tensor = input_tensor.unsqueeze(0)  # 增加 batch 维度
                    input_tensor.requires_grad = True
                    output = net(input_tensor)
                    target_class = output.argmax(dim=1).item()
                    output[0, target_class].backward()
                    grad_cam = input_tensor.grad.data.abs().mean(dim=1).squeeze().cpu().numpy()
                    return grad_cam

                attentions1 = np.array([generate_attention_map(inp) for inp in inputs_selected])
                attentions2 = np.array([generate_attention_map(inp) for inp in inputs[mixup_idx_selected]])

                # 生成 MixUp 数据
                inputs_mixed, labels_mixed = generate_attention_guided_mixup_with_labels(
                    inputs_selected.permute(0, 2, 3, 1).cpu().numpy(),
                    attentions1,
                    labels_selected.cpu().numpy(),
                    inputs[mixup_idx_selected].permute(0, 2, 3, 1).cpu().numpy(),
                    attentions2,
                    labels[mixup_idx_selected].cpu().numpy()
                )

                # 确保 inputs_mixed 数据类型为 torch.float32
                inputs_mixed = torch.tensor(inputs_mixed, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # 转回 [N, C, H, W]

                # 将增强后的 inputs 和 labels 替换进原始数据
                inputs[matched_idx] = inputs_mixed
                labels[matched_idx] = torch.tensor(labels_mixed, dtype=torch.long).to(device)  # 转为整数类型
        elif np.random.rand() < 0.3:  # 剩余 30% 的概率
                transform_basic = transforms.Compose([
                    transforms.RandomRotation(degrees=15),  # 随机旋转
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 随机裁剪并调整大小
                ])
                augmented_inputs = torch.stack([transform_basic(img) for img in inputs.cpu()]).to(device)
                inputs = augmented_inputs

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累计损失
        running_loss += loss.item()
        tqdm_train.set_postfix(loss=loss.item())

    epoch_time = time() - start
    train_loss = running_loss / len(data_loader['train'])
    train_losses.append(train_loss)  # 记录训练损失
    print(f"Epoch time: {epoch_time / 60:.3f} min")
    print(f"Epoch {epoch + 1} Train loss: {train_loss:.3f}")

    # 保存当前 epoch 的模型
    model_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
    torch.save(net.state_dict(), model_save_path)
    print(f"Model saved: {model_save_path}")
    
        # 保存当前 epoch 的模型和状态
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses
    }
    checkpoint_path = os.path.join(save_dir, "checkpoint_epoch_latest.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', linestyle='-', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()

# 保存为 PNG 文件
output_dir = "saved_models/faceswap_cutmix_333"
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，创建文件夹
plot_path = os.path.join(output_dir, "training_loss_curve.png")
plt.savefig(plot_path, dpi=300)  # 保存图形，dpi=300 保证高清
print(f"Training loss curve saved to {plot_path}")