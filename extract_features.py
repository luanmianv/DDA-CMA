import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.swin_transformer import SwinTransformer

# 设置路径
real_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/crossmean/0_features.npy"
fake_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/crossmean/4_features.npy"
model_path = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/saved_models/neuraltextures_cutmix/model_epoch_30.pth"

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SwinTransformer(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=10,  # 和训练时的类别数一致
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
model.load_state_dict(torch.load(model_path))
model.eval()  # 切换到推理模式
model.to(device)

# 提取特征函数
# 提取特征函数
def extract_features(folder_path, model, transform, device):
    video_features = []
    video_names = sorted(os.listdir(folder_path))  # 获取所有视频子文件夹名称
    for video_name in tqdm(video_names, desc=f"Extracting features from {folder_path}"):
        video_path = os.path.join(folder_path, video_name)
        frame_features = []
        frame_names = sorted(os.listdir(video_path))  # 获取所有帧的文件名
        
        # 过滤非图片文件
        frame_names = [f for f in frame_names if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for frame_name in frame_names:
            frame_path = os.path.join(video_path, frame_name)
            # 加载并预处理图片
            try:
                image = Image.open(frame_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                # 提取特征
                with torch.no_grad():
                    feature = model.forward_features(input_tensor)  # 提取最后一层特征
                frame_features.append(feature.cpu().numpy().squeeze())  # 压缩多余的维度
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
                continue
        
        # 确保帧数量一致
        if len(frame_features) != 32:
            print(f"Warning: Video {video_name} has {len(frame_features)} frames, expected 32. Padding or truncating...")
        frame_features = frame_features[:32]  # 截断多余帧
        while len(frame_features) < 32:  # 填充缺失帧
            frame_features.append(np.zeros_like(frame_features[0]))
        # 每个视频的特征形状为 [32, 特征维度]
        video_features.append(np.array(frame_features))
    # 返回所有视频特征 [视频数, 帧数, 特征维度]
    return np.array(video_features, dtype=np.float32)

# 提取 real 和 fake 的特征
real_features = extract_features(real_path, model, data_transform, device)
fake_features = extract_features(fake_path, model, data_transform, device)

# 保存特征到 .npy 文件
output_dir = "/root/autodl-tmp/FACTOR-master/FaceX-Zoo/efficientnet/swin/crossmean"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "1_features.npy"), real_features)
np.save(os.path.join(output_dir, "meiyong_features.npy"), fake_features)
print(f"Features saved to {output_dir}")