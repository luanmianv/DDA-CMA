import cv2
import numpy as np

def generate_mixup(image1, image2, alpha=0.5):
    """
    单纯的 MixUp 图像生成
    """
    mixup_image = alpha * image1 + (1 - alpha) * image2
    return mixup_image.astype(np.uint8)

def generate_attention_guided_mixup_with_labels(images1, attentions1, labels1, images2, attentions2, labels2, alpha=0.5):
    """
    基于注意力引导的 MixUp 图像和标签生成
    """
    # 调整 Attention Map 的形状为 [B, H, W, 1] 并归一化
    attentions1 = (attentions1 - np.min(attentions1, axis=(1, 2), keepdims=True)) / (
        np.max(attentions1, axis=(1, 2), keepdims=True) - np.min(attentions1, axis=(1, 2), keepdims=True) + 1e-8)
    attentions2 = (attentions2 - np.min(attentions2, axis=(1, 2), keepdims=True)) / (
        np.max(attentions2, axis=(1, 2), keepdims=True) - np.min(attentions2, axis=(1, 2), keepdims=True) + 1e-8)

    # 添加通道维度 [B, H, W] -> [B, H, W, 1]
    attentions1 = attentions1[:, :, :, None]
    attentions2 = attentions2[:, :, :, None]

    # 计算权重化的线性组合
    mixup_images = (alpha * attentions1 * images1 + (1 - alpha) * attentions2 * images2) / (
                   alpha * attentions1 + (1 - alpha) * attentions2 + 1e-8)

    # 确保数据类型为 float32
    mixup_images = mixup_images.astype(np.float32)

    # 混合标签
    mixup_labels = alpha * labels1 + (1 - alpha) * labels2

    return mixup_images, mixup_labels