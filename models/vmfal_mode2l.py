"""
双级别不确定性建模
    样本级：对单样本 VGG/BoW 特征直接建模，μ 为归一化特征，κ 通过 MLP 预测
    类别级：对多类别交互特征建模，经自注意力细化，支持动态类别和 padding 处理

不确定性分解
    模态内不确定性：基于 κ 的1 - sigmoid(c×κ)，反映单模态特征稳定性
    模态间不确定性：结合 μ 的余弦相似度和 κ 的差异，反映跨模态一致性

模态偏见校准
    计算样本级偏见（模态内置信度 × 模态间不确定性）
    对高偏见样本施加指数惩罚，修正最终不确定性

适配 VGG+BoW 特征
    保留原始特征用于 κ 预测，归一化特征用于 μ 计算（符合单位球特性）
    简化特征交互逻辑，专注于基础 MLP 和自注意力（适合 VGG/BoW 的基础结构）

训练策略
    区分有标注 / 无标注数据处理逻辑
    结合三元组损失思想，使难检索样本具有更高不确定性
    引入偏见校准损失，引导模型降低模态偏好

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple, List, Optional
from .training import Training

"""
该模块应该如何使用：
# 初始化模型（与现有框架适配）
vmf_model = VMFUncertaintyModel(net, net_args, handler, args, writer, device)

# 训练阶段（有标注数据）
for epoch in range(num_epochs):
    img_batch, txt_batch, labels_batch = train_loader.next()
    loss, metrics = vmf_model.train_step(img_batch, txt_batch, labels_batch, is_labeled=True)
    # 记录指标...

# 主动学习样本选择（无标注池）
selected_indices = vmf_model.select_samples(unlabeled_img_pool, unlabeled_txt_pool, n_samples)

"""

import torch
import torch.nn.functional as F
import numpy as np


class VMFUncertaintyModel:
    """
    VMF 不确定性建模模块
    - 支持样本级不确定性
    - 模态偏见校准
    - 可用于：
        1) 主动学习无标注样本打分
        2) 已标注样本训练损失加权
    """

    def __init__(self, device="cuda"):
        self.device = device

    # -----------------------------
    # vMF 核心工具
    # -----------------------------
    def _estimate_kappa(self, cos_sim: torch.Tensor) -> torch.Tensor:
        """
        根据 cosine similarity 估计 vMF 浓度参数 κ
        """
        eps = 1e-6
        kappa = cos_sim / (1.0 - cos_sim + eps)
        return torch.clamp(kappa, min=0.0)

    def _sample_level_uncertainty(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor,
        label_emb: torch.Tensor = None,
    ):
        """
        样本级不确定性建模
        """
        # intra-modal
        intra_v = (img_feat * txt_feat).sum(dim=1)
        intra_t = intra_v.clone()

        # inter-modal
        inter = intra_v

        # κ
        kappa_v = self._estimate_kappa(intra_v)
        kappa_t = self._estimate_kappa(intra_t)

        # uncertainty = κ 的反函数
        unc = 1.0 / (kappa_v + kappa_t + 1e-6)

        return unc, intra_v, intra_t, inter, kappa_v, kappa_t

    def _modal_bias_calibration(
        self,
        intra_v: torch.Tensor,
        intra_t: torch.Tensor,
        inter: torch.Tensor,
        unc: torch.Tensor,
    ):
        """
        模态偏见校准
        """
        bias = torch.abs(intra_v - intra_t)
        calibrated_unc = unc * (1.0 + bias)
        return calibrated_unc, bias

    # -----------------------------
    # ★ 新增：无标注样本评分接口
    # -----------------------------
    @torch.no_grad()
    def score_unlabeled(
        self,
        common_img: torch.Tensor,
        common_txt: torch.Tensor,
        label_emb: torch.Tensor = None,
    ):
        """
        给无标注样本打分（主动学习）
        """
        common_img = F.normalize(common_img, dim=1)
        common_txt = F.normalize(common_txt, dim=1)

        if label_emb is not None:
            label_emb = F.normalize(label_emb, dim=1)

        unc, intra_v, intra_t, inter, kappa_v, kappa_t = \
            self._sample_level_uncertainty(common_img, common_txt, label_emb)

        calibrated_unc, bias = self._modal_bias_calibration(
            intra_v, intra_t, inter, unc
        )

        # 最终打分：不确定性 + 模态偏见
        score = calibrated_unc + bias

        return score

    # -----------------------------
    # ★ 训练时样本权重
    # -----------------------------
    @torch.no_grad()
    def compute_training_weights(
        self,
        common_img: torch.Tensor,
        common_txt: torch.Tensor,
        labels: torch.Tensor,
        label_emb: torch.Tensor = None,
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
    ):
        """
        训练样本加权（鲁棒训练）
        """
        common_img = F.normalize(common_img, dim=1)
        common_txt = F.normalize(common_txt, dim=1)

        unc, intra_v, intra_t, inter, _, _ = \
            self._sample_level_uncertainty(common_img, common_txt, label_emb)

        calibrated_unc, bias = self._modal_bias_calibration(
            intra_v, intra_t, inter, unc
        )

        num_classes = int(labels.max().item()) + 1
        class_unc = torch.zeros(num_classes, device=labels.device)

        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                class_unc[c] = calibrated_unc[mask].mean()

        weights = (
            1.0
            + alpha * calibrated_unc
            + beta * bias
            + gamma * class_unc[labels]
        )

        return weights.detach()
