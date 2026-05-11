"""
VLMo4CMRModel: 受 VLMo (Bao et al., ICLR 2022) 启发的跨模态检索骨干

VLMo 核心设计理念 (适配到特征级融合):
1. 共享多头注意力 (Shared Multi-Head Attention): 两个模态共享同一组 Attention 权重
2. 模态混合专家 (Mixture-of-Modality-Experts, MoME):
   每层 Transformer 的 FFN 替换为模态特异的专家网络
3. 统一架构: 同一模型同时支持单模态理解和跨模态融合

原论文: "VLMo: Unified Vision-Language Pre-Training with
         Mixture-of-Modality-Experts" (ICLR 2022)

适配说明:
- 原论文: 多 token 序列 → 多层 Transformer (共享 Attn + MoME FFN)
- 本实现: 单向量输入 → 通过 [CLS] + 可学习 context tokens 构造序列
  这样 Self-Attention 才有意义（而不是退化为恒等映射）
- 保留核心: 共享 Attention + 模态特异 FFN

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


class MoME_FFN(nn.Module):
    """
    VLMo 的模态混合专家 FFN (Mixture-of-Modality-Experts)。

    原论文: 每层 Transformer Block 共享 Self-Attention，但 FFN 部分
    根据输入模态路由到不同的专家网络。
    """

    def __init__(self, dim, ffn_ratio=4, dropout=0.1):
        super().__init__()
        hidden = dim * ffn_ratio
        # 图像专家
        self.img_expert = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        # 文本专家
        self.txt_expert = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
        # 融合专家（VLMo 在跨模态任务中使用第三个专家）
        self.fusion_expert = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, modality='image'):
        if modality == 'image':
            return self.img_expert(x)
        elif modality == 'text':
            return self.txt_expert(x)
        else:  # 'fusion'
            return self.fusion_expert(x)


class VLMoTransformerBlock(nn.Module):
    """
    VLMo 风格的 Transformer Block:
    共享 Self-Attention + 模态特异 MoME FFN
    """

    def __init__(self, dim, num_heads=4, dropout=0.1, ffn_ratio=4):
        super().__init__()
        # 共享的多头自注意力（VLMo 核心特性 1）
        self.shared_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        # MoME: 模态特异 FFN（VLMo 核心特性 2）
        self.mome = MoME_FFN(dim, ffn_ratio=ffn_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, modality='image'):
        """
        x: [B, S, D]  (S = 1 + n_context_tokens)
        modality: 'image' / 'text' / 'fusion'
        """
        # 共享 Self-Attention
        attn_out, _ = self.shared_attn(x, x, x)
        x = self.norm1(x + attn_out)
        # 模态特异 FFN
        ffn_out = self.mome(x, modality=modality)
        x = self.norm2(x + ffn_out)
        return x


class VLMo4CMRModel(nn.Module):
    """
    VLMo 风格的跨模态检索模型（特征级适配版）。

    架构:
    ┌────────────────────────────────────────┐
    │     Input Mapping + Context Tokens      │
    │  img_dim→D, txt_dim→D + learnable ctx  │
    └────────────┬──────────────┬────────────┘
                 │              │
    ┌────────────▼──────────────▼────────────┐
    │    VLMo Transformer Blocks (×N)         │
    │    共享 Self-Attention                   │
    │    + MoME FFN (img/txt/fusion 专家)     │
    │                                          │
    │    Phase 1: 各模态独立过 (img/txt 专家)  │
    │    Phase 2: 拼接后过 (fusion 专家)       │
    └────────────┬──────────────┬────────────┘
                 │              │
    ┌────────────▼──┐   ┌──────▼────────────┐
    │   img_pred     │   │   text_pred       │
    └───────────────┘   └───────────────────┘
    """

    def __init__(self, num_class, img_dim=512, text_dim=512, mid_dim=512,
                 feature_dim=1024, dropout_prob=0.1, init_weight=True,
                 n_layers=2, n_heads=4, n_context=4):
        super().__init__()

        # 输入映射
        self.img_map = nn.Sequential(
            nn.Linear(img_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.txt_map = nn.Sequential(
            nn.Linear(text_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # 可学习的 context tokens
        # 原论文处理 token 序列，我们只有 1 个向量
        # 添加 context tokens 使 Self-Attention 有多个 token 可 attend
        self.n_context = n_context
        if n_context > 0:
            self.img_context = nn.Parameter(torch.randn(1, n_context, feature_dim) * 0.02)
            self.txt_context = nn.Parameter(torch.randn(1, n_context, feature_dim) * 0.02)

        # 共享 Transformer Blocks (共享 Attention + MoME FFN)
        self.blocks = nn.ModuleList([
            VLMoTransformerBlock(feature_dim, num_heads=n_heads,
                                 dropout=dropout_prob, ffn_ratio=4)
            for _ in range(n_layers)
        ])

        # 可选：额外的融合层（VLMo 在跨模态任务中拼接两模态 token 序列）
        self.fusion_block = VLMoTransformerBlock(
            feature_dim, num_heads=n_heads, dropout=dropout_prob, ffn_ratio=4
        )

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
        """
        B = img_features.shape[0]

        # --- Phase 1: 映射 + 构造序列 ---
        img_tok = self.img_map(img_features.float()).unsqueeze(1)  # [B, 1, D]
        txt_tok = self.txt_map(text_features.float()).unsqueeze(1)  # [B, 1, D]

        if self.n_context > 0:
            img_ctx = self.img_context.expand(B, -1, -1)  # [B, n_ctx, D]
            txt_ctx = self.txt_context.expand(B, -1, -1)  # [B, n_ctx, D]
            img_seq = torch.cat([img_tok, img_ctx], dim=1)  # [B, 1+n_ctx, D]
            txt_seq = torch.cat([txt_tok, txt_ctx], dim=1)  # [B, 1+n_ctx, D]
        else:
            img_seq = img_tok
            txt_seq = txt_tok

        # --- Phase 2: 各模态独立过共享 Transformer (用各自的 MoME 专家) ---
        for block in self.blocks:
            img_seq = block(img_seq, modality='image')
            txt_seq = block(txt_seq, modality='text')

        # --- Phase 3: 跨模态融合 (VLMo fusion mode) ---
        # 拼接两模态的 token 序列，用 fusion 专家
        combined = torch.cat([img_seq, txt_seq], dim=1)  # [B, 2*(1+n_ctx), D]
        fused = self.fusion_block(combined, modality='fusion')

        # 取各模态的第一个 token (CLS position) 作为输出
        S = img_seq.shape[1]  # 1 + n_context
        img_out = fused[:, 0, :]  # [B, D] - img CLS
        txt_out = fused[:, S, :]  # [B, D] - txt CLS

        # L2 归一化
        img_features_out = l2norm(img_out, dim=1)
        text_features_out = l2norm(txt_out, dim=1)

        # 预测
        w_norm = F.normalize(self.predictLayer.weight, dim=1)
        centers_norm = F.normalize(self.centers, dim=0)

        img_pred = F.linear(img_features_out, w_norm, self.predictLayer.bias)
        text_pred = F.linear(text_features_out, w_norm, self.predictLayer.bias)

        return centers_norm, img_features_out, text_features_out, img_pred, text_pred