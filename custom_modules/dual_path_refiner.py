import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LightweightMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        reduction_ratio: int = 4,
        window_size: int = 7
    ):
        super(LightweightMultiHeadSelfAttention, self).__init__()
        assert reduction_ratio > 0
        reduced_channels = max(num_heads, (in_channels // reduction_ratio) // num_heads * num_heads)
        self.in_channels = in_channels
        self.reduced_channels = reduced_channels
        self.num_heads = num_heads
        self.head_dim = self.reduced_channels // self.num_heads
        self.window_size = window_size

        self.qkv_conv = nn.Conv2d(in_channels, reduced_channels * 3, kernel_size=1, bias=False)
        self.proj_conv = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        identity = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        qkv = self.qkv_conv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        window_size = self.window_size
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        if pad_r > 0 or pad_b > 0:
            q = F.pad(q, (0, pad_r, 0, pad_b))
            k = F.pad(k, (0, pad_r, 0, pad_b))
            v = F.pad(v, (0, pad_r, 0, pad_b))
        _, _, Hp, Wp = q.shape

        num_windows_h = Hp // window_size
        num_windows_w = Wp // window_size
        q = q.view(B, self.num_heads, self.head_dim, Hp, Wp)
        k = k.view(B, self.num_heads, self.head_dim, Hp, Wp)
        v = v.view(B, self.num_heads, self.head_dim, Hp, Wp)
        q = q.view(B, self.num_heads, self.head_dim, num_windows_h, window_size, num_windows_w, window_size)
        k = k.view(B, self.num_heads, self.head_dim, num_windows_h, window_size, num_windows_w, window_size)
        v = v.view(B, self.num_heads, self.head_dim, num_windows_h, window_size, num_windows_w, window_size)
        q = q.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        k = k.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        v = v.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        q = q.view(B * self.num_heads * num_windows_h * num_windows_w, self.head_dim, window_size * window_size)
        k = k.view(B * self.num_heads * num_windows_h * num_windows_w, self.head_dim, window_size * window_size)
        v = v.view(B * self.num_heads * num_windows_h * num_windows_w, self.head_dim, window_size * window_size)

        scale = (self.head_dim) ** -0.5
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)

        out = out.view(B, self.num_heads, num_windows_h, num_windows_w, self.head_dim, window_size, window_size)
        out = out.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        out = out.view(B, self.reduced_channels, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W]
        out = self.proj_conv(out)
        out = self.dropout(out)
        out = out + identity

        identity2 = out
        out_norm = self.norm2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.ffn(out_norm)
        out = out + identity2
        return out


class DualPathRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        reduction_ratio: int = 4,
        use_residual: bool = True,
        use_lightweight_attn: bool = True,
        window_size: int = 7
    ):
        super(DualPathRefiner, self).__init__()
        self.in_channels = in_channels
        self.use_residual = use_residual
        self.use_lightweight_attn = use_lightweight_attn

        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        if use_lightweight_attn:
            self.attn_branch = LightweightMultiHeadSelfAttention(
                in_channels=in_channels,
                num_heads=num_heads,
                dropout=dropout,
                reduction_ratio=reduction_ratio,
                window_size=window_size
            )
        else:
            self.attn_branch = None

        self.fusion_weight = nn.Parameter(torch.ones(2) * 0.5)
        self.output_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        conv_feat = self.conv_branch(x)
        if self.attn_branch is not None:
            attn_feat = self.attn_branch(x)
            fusion_weights = F.softmax(self.fusion_weight, dim=0)
            fused_feat = fusion_weights[0] * conv_feat + fusion_weights[1] * attn_feat
        else:
            fused_feat = conv_feat
        out = self.output_proj(fused_feat)
        if self.use_residual:
            out = out + identity
        return out


class DualPathFPNRefinement(nn.Module):
    def __init__(
        self,
        in_channels: int,
        fpn_levels: list = ['p2', 'p3', 'p4', 'p5', 'p6'],
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        reduction_ratio: int = 4,
        enabled_levels: Optional[list] = None,
        low_levels_conv_only: bool = True
    ):
        super(DualPathFPNRefinement, self).__init__()
        self.fpn_levels = fpn_levels
        self.enabled_levels = enabled_levels if enabled_levels is not None else fpn_levels
        self.low_levels_conv_only = low_levels_conv_only
        low_levels = ['p2', 'p3']

        self.refinement_modules = nn.ModuleDict()
        for level in fpn_levels:
            if level in self.enabled_levels:
                if low_levels_conv_only and level in low_levels:
                    self.refinement_modules[level] = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  padding=kernel_size // 2, groups=1, bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_channels)
                    )
                else:
                    self.refinement_modules[level] = DualPathRefiner(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                        num_heads=num_heads,
                        dropout=dropout,
                        reduction_ratio=reduction_ratio,
                        use_residual=True,
                        use_lightweight_attn=True,
                        window_size=7
                    )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enhanced_features = {}
        for level, feat in features.items():
            if level in self.refinement_modules:
                enhanced = self.refinement_modules[level](feat)
                if self.low_levels_conv_only and level in ['p2', 'p3']:
                    enhanced = enhanced + feat
                enhanced_features[level] = enhanced
            else:
                enhanced_features[level] = feat
        return enhanced_features


