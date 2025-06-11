import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image

class DynamicCutMix:
    def __init__(self, model, target_layer, device):
        """
        初始化 DynamicCutMix 实例
        :param model: 用于训练的模型
        :param target_layer: 用于 Grad-CAM 的目标层
        :param device: 运行设备
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

        # 初始化 Hook 存储
        self.feature_maps = None
        self.gradients = None

        # 注册 Hook
        self._register_hooks()

    def _register_hooks(self):
        """
        注册前向和后向钩子函数，用于捕获特征图和梯度
        """
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.feature_maps = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _generate_grad_cam(self, input_tensor, target_class):
        """
        动态生成 Grad-CAM
        :param input_tensor: 输入图像的 Tensor
        :param target_class: 目标类别索引
        :return: 归一化的 Grad-CAM 热力图
        """
        self.model.eval()  # 切换到 eval 模式
        self.feature_maps = None
        self.gradients = None

        # 前向传播
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        output = self.model(input_tensor)

        # 获取目标类别分数
        target_score = output[0, target_class]

        # 反向传播计算梯度
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Reshape 特征图
        batch_size, num_patches, embed_dim = self.feature_maps.shape
        height = width = int(num_patches ** 0.5)  # 假设特征图是正方形
        feature_maps_reshaped = self.feature_maps.view(batch_size, height, width, embed_dim)

        # 计算梯度权重
        weights = torch.mean(self.gradients, dim=[0, 1])

        # 生成 Grad-CAM
        grad_cam = torch.zeros((height, width), device=self.device)
        for i, w in enumerate(weights):
            grad_cam += w * feature_maps_reshaped[0, :, :, i]

        grad_cam = F.relu(grad_cam).cpu().detach().numpy()

        # 归一化
        grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam) + 1e-8)
        return grad_cam

    def _generate_cutmix_image(self, image1, attention1, image2, attention2):
        # 调整 Attention Map 到输入图片大小
        attention1_resized = cv2.resize(attention1, (image1.shape[1], image1.shape[0]))
        attention2_resized = cv2.resize(attention2, (image2.shape[1], image2.shape[0]))

        attention1_resized = (attention1_resized - np.min(attention1_resized)) / (np.max(attention1_resized) - np.min(attention1_resized) + 1e-8)
        attention2_resized = (attention2_resized - np.min(attention2_resized)) / (np.max(attention2_resized) - np.min(attention2_resized) + 1e-8)

        # 使用 Otsu 分割生成二值掩码
        _, mask = cv2.threshold((attention1_resized * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 连通区域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 检查是否有有效区域
        if num_labels <= 1:  # 没有前景区域
            print("No valid region found, returning original image1.")
            return image1  # 返回原始图像



        # 获取最大连通区域的外接矩形
        largest_area_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        x, y, w, h = stats[largest_area_idx, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]

        # 创建矩形掩码
        rect_mask = np.zeros_like(mask, dtype=np.uint8)
        rect_mask[y:y + h, x:x + w] = 1

        # 生成 CutMix 图像
        cutmix_image = rect_mask[:, :, None] * attention1_resized[:, :, None] * image1 + \
                   (1 - rect_mask[:, :, None] * attention1_resized[:, :, None]) * image2
        return cutmix_image.astype(np.uint8)
    
    def augment(self, tensor1, tensor2, target_class):
        """
        执行 CutMix 数据增强
        :param tensor1: 第一张输入图像的 Tensor
        :param tensor2: 第二张输入图像的 Tensor
        :param target_class: 目标类别索引
        :return: 增强后的 CutMix 图像的 Tensor
        """
        # 生成 Grad-CAM 热力图
        grad_cam1 = self._generate_grad_cam(tensor1, target_class)
        grad_cam2 = self._generate_grad_cam(tensor2, target_class)

        # 将 Tensor 转为 numpy array
        image1_np = tensor1.permute(1, 2, 0).cpu().numpy() * 255
        image2_np = tensor2.permute(1, 2, 0).cpu().numpy() * 255

        # 生成 CutMix 图像
        cutmix_image = self._generate_cutmix_image(image1_np.astype(np.uint8), grad_cam1,
                                                   image2_np.astype(np.uint8), grad_cam2)

        # 转回 Tensor
        cutmix_tensor = transforms.ToTensor()(Image.fromarray(cutmix_image.astype(np.uint8)))
        return cutmix_tensor
    
    
    
    
    