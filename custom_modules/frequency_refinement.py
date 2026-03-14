"""
Frequency-Guided Feature Refinement Module
频域引导的高低频特征增强模块

核心思想：
- 水下图像受散射导致纹理模糊、细节损失
- 在FPN或主干输出后，分离并强化高频纹理信息、抑制低频色偏干扰
- 通过AvgPool分离低频，通过残差提取高频，通过卷积增强高频特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FrequencyGuidedRefinement(nn.Module):
    """
    频域引导特征增强模块
    
    实现方式：
    1. 低频：F_l = AvgPool(X) - 通过平均池化提取低频信息
    2. 高频：F_h = X - F_l - 通过残差提取高频信息
    3. 增强输出：X' = X + Conv(F_h) - 高频残差叠加
    4. 可选：通道门控（Sigmoid）自适应调整权重
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        use_gate: bool = True,
        reduction: int = 16
    ):
        """
        Args:
            in_channels: 输入特征通道数
            kernel_size: 卷积核大小，用于高频特征增强
            use_gate: 是否使用通道门控机制
            reduction: 门控机制的通道压缩比例
        """
        super(FrequencyGuidedRefinement, self).__init__()
        
        self.in_channels = in_channels
        self.use_gate = use_gate
        
        # 高频特征增强卷积层
        # 使用3x3卷积来增强高频纹理信息
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                     padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # 通道门控机制（可选）
        if use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.gate = None
            
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模块权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            enhanced_x: 增强后的特征图 [B, C, H, W]
        """
        # 1. 提取低频信息：F_l = AvgPool(X)
        # 使用平均池化提取低频信息，池化核大小自适应特征图尺寸
        # 对于不同尺度的特征图，使用合适的池化核大小
        h, w = x.shape[2], x.shape[3]
        
        # 计算合适的池化核大小（约为特征图尺寸的1/4到1/8，但至少为3）
        pool_size_h = max(3, min(h // 4, 7))
        pool_size_w = max(3, min(w // 4, 7))
        
        # 如果特征图太小，使用全局平均池化
        if h <= pool_size_h or w <= pool_size_w:
            # 全局平均池化 + 扩展回原尺寸
            low_freq = F.adaptive_avg_pool2d(x, 1)
            low_freq = F.interpolate(low_freq, size=(h, w), mode='bilinear', align_corners=False)
        else:
            # 使用局部平均池化提取低频，然后上采样回原尺寸
            low_freq = F.avg_pool2d(x, kernel_size=(pool_size_h, pool_size_w), 
                                   stride=1, padding=(pool_size_h//2, pool_size_w//2))
            # 确保尺寸匹配（可能因为padding导致尺寸略有差异）
            if low_freq.shape[2] != h or low_freq.shape[3] != w:
                low_freq = F.interpolate(low_freq, size=(h, w), mode='bilinear', align_corners=False)
        
        # 2. 提取高频信息：F_h = X - F_l
        high_freq = x - low_freq
        
        # 3. 对高频特征进行卷积增强：Conv(F_h)
        enhanced_high_freq = self.high_freq_conv(high_freq)
        
        # 4. 应用通道门控（如果启用）：自适应调整权重
        if self.use_gate and self.gate is not None:
            gate_weight = self.gate(x)
            enhanced_high_freq = enhanced_high_freq * gate_weight
        
        # 5. 高频残差叠加：X' = X + Conv(F_h)
        enhanced_x = x + enhanced_high_freq
        
        return enhanced_x


class FrequencyGuidedFPNRefinement(nn.Module):
    """
    针对FPN多尺度特征的频域增强模块
    
    对FPN的每个层级（p2, p3, p4, p5, p6）分别应用频域增强
    """
    
    def __init__(
        self,
        in_channels: int,
        fpn_levels: list = ['p2', 'p3', 'p4', 'p5', 'p6'],
        use_gate: bool = True,
        enabled_levels: Optional[list] = None
    ):
        """
        Args:
            in_channels: FPN特征通道数（通常为256）
            fpn_levels: FPN层级列表
            use_gate: 是否使用通道门控
            enabled_levels: 启用增强的层级列表，None表示全部启用
        """
        super(FrequencyGuidedFPNRefinement, self).__init__()
        
        self.fpn_levels = fpn_levels
        self.enabled_levels = enabled_levels if enabled_levels is not None else fpn_levels
        
        # 为每个FPN层级创建独立的增强模块
        self.refinement_modules = nn.ModuleDict()
        for level in fpn_levels:
            if level in self.enabled_levels:
                self.refinement_modules[level] = FrequencyGuidedRefinement(
                    in_channels=in_channels,
                    kernel_size=3,
                    use_gate=use_gate
                )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: FPN特征字典，key为层级名（如'p2', 'p3'等），value为特征图
        
        Returns:
            enhanced_features: 增强后的FPN特征字典
        """
        enhanced_features = {}
        
        for level, feat in features.items():
            if level in self.refinement_modules:
                # 应用频域增强
                enhanced_features[level] = self.refinement_modules[level](feat)
            else:
                # 不增强，直接传递
                enhanced_features[level] = feat
        
        return enhanced_features

