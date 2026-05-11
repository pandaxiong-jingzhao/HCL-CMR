import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(X: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """与你上传的 CLIP4CMR model.py 一致的 L2-normalize。"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return torch.div(X, norm)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """与你上传的 CLIP4CMR model.py 一致的 GELU (erf 版本)。"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ImgNN(nn.Module):
    """Network to learn image representations (与原 repo 对齐)"""
    def __init__(self, input_dim=512, mindum_dim=512, out_dim=1024, dropout_prob=0.1):
        super().__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class TextNN(nn.Module):
    """Network to learn text representations (与原 repo 对齐)"""
    def __init__(self, input_dim=512, mindum_dim=512, out_dim=1024, dropout_prob=0.1):
        super().__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class CLIP4CMRModel(nn.Module):
    """
    CLIP4CMR 主体模型（严格对齐你上传的 model.py: class model）。
    forward 返回：
        centers, img_features, text_features, img_pred, text_pred
    """

    def __init__(self, num_class, img_dim=512, text_dim=512, mid_dim=1024,
                 feature_dim=512, dropout_prob=0.1, init_weight=True,
                 use_attention=True):  # 新增参数
        super().__init__()

        # 1. 增加网络深度和宽度
        self.imgnn = nn.Sequential(
            nn.Linear(img_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, feature_dim)
        )

        self.textnn = nn.Sequential(
            nn.Linear(text_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mid_dim, feature_dim)
        )

        # 2. 新增跨模态注意力层（可选）
        self.use_attention = use_attention
        if use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                dropout=dropout_prob,
                batch_first=True
            )

        self.n_classes = num_class
        self.feat_dim = feature_dim

        # 原 repo：predictLayer 有 bias=True
        self.predictLayer = nn.Linear(self.feat_dim, self.n_classes, bias=True)

        # 原 repo：centers requires_grad=False
        #self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=False)
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=True)

        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        # 原 repo：kaiming_normal_ 初始化 centers 与 predictLayer.weight
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def get_embedding_dim(self) -> int:
        return self.feat_dim

    # def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
    #     # 1) 输入是外部预提取特征
    #     img_features = l2norm(self.imgnn(img_features.float()), dim=1)
    #     text_features = l2norm(self.textnn(text_features.float()), dim=1)
    #
    #     # 2) 计算归一化权重，并更新模型参数（保持一致性）
    #     with torch.no_grad():
    #         # 归一化并写回参数（训练过程中保持归一化）
    #         self.predictLayer.weight.data = l2norm(self.predictLayer.weight.data, dim=-1)
    #         self.centers.data = l2norm(self.centers.data, dim=0)
    #
    #     # 3) 直接使用更新后的参数
    #     img_pred = self.predictLayer(img_features)
    #     text_pred = self.predictLayer(text_features)
    #
    #     # 4) 返回用于loss的centers
    #     return self.centers, img_features, text_features, img_pred, text_pred
    # def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
    #     # 1) 输入是外部预提取特征
    #     img_features = l2norm(self.imgnn(img_features.float()), dim=1)
    #     text_features = l2norm(self.textnn(text_features.float()), dim=1)
    #
    #     # 2) 关键修复：不要 .data 写回；只在计算时使用归一化副本（梯度可回传）
    #     # predictLayer.weight 形状通常是 [C, D]，所以对每个类别权重向量归一化：dim=1
    #     w_norm = F.normalize(self.predictLayer.weight, dim=1)
    #
    #     # centers 形状是 [D, C]（你当前实现），对每个类别中心向量归一化：dim=0（按列）
    #     centers_norm = F.normalize(self.centers, dim=0)
    #
    #     # 3) 用归一化后的权重计算 logits（避免 self.predictLayer(...) 用到未归一化 weight）
    #     img_pred = F.linear(img_features, w_norm,self.predictLayer.bias)  # [B,D] x [C,D]^T -> [B,C]
    #     text_pred = F.linear(text_features, w_norm,self.predictLayer.bias)
    #
    #     # 4) 返回给 loss 用的 centers：返回 centers_norm（稳定且符合球面/vMF几何）
    #     return centers_norm, img_features, text_features, img_pred, text_pred

    def forward(self, img_features: torch.Tensor, text_features: torch.Tensor):
        # 1) 输入是外部预提取特征
        img_hidden = self.imgnn(img_features.float())
        text_hidden = self.textnn(text_features.float())

        # 2) 应用跨模态注意力（如果启用）
        if self.use_attention:
            # 将两个模态的特征堆叠为序列 [batch_size, 2, feature_dim]
            combined = torch.stack([img_hidden, text_hidden], dim=1)

            # 应用多头注意力
            # 注意：多头注意力需要输入形状 [batch_size, seq_len, feature_dim]
            attended, _ = self.cross_attention(combined, combined, combined)

            # 分离回两个模态
            img_hidden = attended[:, 0, :]  # 取第一个位置（图像特征）
            text_hidden = attended[:, 1, :]  # 取第二个位置（文本特征）

        # 3) 归一化
        img_features_out = l2norm(img_hidden, dim=1)
        text_features_out = l2norm(text_hidden, dim=1)

        # 4) 关键修复：不要 .data 写回；只在计算时使用归一化副本（梯度可回传）
        # predictLayer.weight 形状通常是 [C, D]，所以对每个类别权重向量归一化：dim=1
        w_norm = F.normalize(self.predictLayer.weight, dim=1)

        # centers 形状是 [D, C]（你当前实现），对每个类别中心向量归一化：dim=0（按列）
        centers_norm = F.normalize(self.centers, dim=0)

        # 5) 用归一化后的权重计算 logits
        img_pred = F.linear(img_features_out, w_norm, self.predictLayer.bias)  # [B,D] x [C,D]^T -> [B,C]
        text_pred = F.linear(text_features_out, w_norm, self.predictLayer.bias)

        # 6) 返回给 loss 用的 centers
        return centers_norm, img_features_out, text_features_out, img_pred, text_pred

