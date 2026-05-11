"""
ALBEF4CMRModel: 受 ALBEF (Li et al., NeurIPS 2021) 启发的跨模态检索骨干

ALBEF 核心设计理念 (适配到特征级融合):
1. Align: 各模态独立投影到共享空间 → 对比学习对齐
2. Fuse:  Cross-Attention 深度融合 → 用于分类/匹配

原论文: "Align before Fuse: Vision and Language Representation Learning
         with Momentum Distillation" (NeurIPS 2021)

适配说明:
- 原论文输入: 原始图像(ViT patch tokens) + 原始文本(BERT word tokens)
- 本实现输入: CLIP 预提取的 512-d 向量
- 保留核心思想: 先独立投影对齐(Align)，再交叉注意力融合(Fuse)
- Cross-Attention: 图像 attend 文本 / 文本 attend 图像（双向）
- 多层 Transformer Block 模拟原论文的深度融合

forward 返回与 CLIP4CMRModel 完全一致:
    centers, img_features, text_features, img_pred, text_pred
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(X: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(X, norm)


class CrossAttentionBlock(nn.Module):
    """
    ALBEF 风格的 Cross-Attention Block:
    - Query 来自一个模态，Key/Value 来自另一个模态
    - 后接 LayerNorm + FFN + LayerNorm (Post-LN Transformer)
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        """
        query:     [B, 1, D] 或 [B, S_q, D]
        key_value: [B, 1, D] 或 [B, S_kv, D]
        """
        # Cross-Attention: Q 来自 query 模态, K/V 来自 key_value 模态
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ALBEF4CMRModel(nn.Module):
    """
    ALBEF 风格的跨模态检索模型（特征级适配版）。

    架构:
    ┌─────────────┐    ┌─────────────┐
    │  Image Proj  │    │  Text Proj   │     ← Align Phase
    │  (3层 MLP)   │    │  (3层 MLP)   │
    └──────┬──────┘    └──────┬──────┘
           │                   │
    ┌──────▼──────────────────▼──────┐
    │   Cross-Attention Blocks (×N)   │     ← Fuse Phase
    │   img←txt / txt←img 双向交叉    │
    └──────┬──────────────────┬──────┘
           │                   │
    ┌──────▼──────┐    ┌──────▼──────┐
    │  img_pred    │    │  text_pred   │
    └─────────────┘    └─────────────┘
    """

    def __init__(self, num_class, img_dim=512, text_dim=512, mid_dim=512,
                 feature_dim=1024, dropout_prob=0.1, init_weight=True,
                 n_cross_layers=2, n_heads=4):
        super().__init__()

        # === Align Phase: 独立模态投影 ===
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, feature_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, feature_dim),
        )

        # === Fuse Phase: 双向 Cross-Attention ===
        # 图像 attend 文本（img_q, txt_kv）
        self.img_cross_layers = nn.ModuleList([
            CrossAttentionBlock(feature_dim, num_heads=n_heads, dropout=dropout_prob)
            for _ in range(n_cross_layers)
        ])
        # 文本 attend 图像（txt_q, img_kv）
        self.txt_cross_layers = nn.ModuleList([
            CrossAttentionBlock(feature_dim, num_heads=n_heads, dropout=dropout_prob)
            for _ in range(n_cross_layers)
        ])

        self.n_classes = num_class
        self.feat_dim = feature_dim

        # 分类头
        self.predictLayer = nn.Linear(feature_dim, num_class, bias=True)
        self.centers = nn.Parameter(torch.randn(feature_dim, num_class), requires_grad=True)

        if init_weight:
            self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def get_embedding_dim(self) -> int:
        return self.feat_dim

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        """
        输入: img_features [B, img_dim], text_features [B, text_dim]
        输出: centers, img_features_out, text_features_out, img_pred, text_pred
              (与 CLIP4CMRModel 完全一致的接口)
        """
        # --- Align: 独立投影 ---
        img_h = self.img_proj(img_features.float())   # [B, D]
        txt_h = self.text_proj(text_features.float())  # [B, D]

        # 添加 seq 维度用于 Attention: [B, D] → [B, 1, D]
        img_seq = img_h.unsqueeze(1)
        txt_seq = txt_h.unsqueeze(1)

        # --- Fuse: 双向 Cross-Attention ---
        # ALBEF 核心: 图像 attend 文本信息，文本 attend 图像信息
        for img_layer, txt_layer in zip(self.img_cross_layers, self.txt_cross_layers):
            # 保存上一步的值，避免信息泄露
            img_prev = img_seq
            txt_prev = txt_seq
            img_seq = img_layer(query=img_prev, key_value=txt_prev)  # img ← txt
            txt_seq = txt_layer(query=txt_prev, key_value=img_prev)  # txt ← img

        # 去掉 seq 维度: [B, 1, D] → [B, D]
        img_fused = img_seq.squeeze(1)
        txt_fused = txt_seq.squeeze(1)

        # L2 归一化
        img_features_out = l2norm(img_fused, dim=1)
        text_features_out = l2norm(txt_fused, dim=1)

        # 归一化权重
        w_norm = F.normalize(self.predictLayer.weight, dim=1)
        centers_norm = F.normalize(self.centers, dim=0)

        # 预测
        img_pred = F.linear(img_features_out, w_norm, self.predictLayer.bias)
        text_pred = F.linear(text_features_out, w_norm, self.predictLayer.bias)

        return centers_norm, img_features_out, text_features_out, img_pred, text_pred