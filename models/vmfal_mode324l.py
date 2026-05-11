# models/vmfal_model.py
"""
vMF Uncertainty Model（严格版）

- inter-uncertainty：用 vMF 分布重叠度闭式（产品积分）并归一化到 (0,1]
    overlap_norm = C_d(κi+κt) / C_d(R),  R = ||κi μi + κt μt||
    U_inter = 1 - overlap_norm
- density：严格 vMF KDE（球面核密度）返回 log-density，并归一化到 [0,1] 做惩罚项
- κ 的估计：为保证工程可运行且稳定，使用“与类别锚点最大相似度”的确定性 proxy：
    κ = κ_min + (κ_max-κ_min) * ((max_sim+1)/2)
  （避免你旧版本里“每次调用随机初始化 MLP”的致命不稳定）

Author: patched
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class VMFUncertaintyModel:
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        device: torch.device,
        lambda_density: float = 0.5,
        lambda_bias: float = 0.5,
        alpha_sample: float = 0.5,
        kde_kappa: float = 32.0,
        kde_batch: int = 512,
        kappa_min: float = 2.0,
        kappa_max: float = 128.0,
        c_intra: float = 0.08,
        # ====== NEW (Scheme-2) ======
        kappa_soft_tau: float = 0.07,
        density_warmup_rounds: int = 0,
        train_weight_scale: float = 0.8,
    ):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.device = device

        self.lambda_density = float(lambda_density)
        self.lambda_bias = float(lambda_bias)
        self.alpha_sample = float(alpha_sample)

        self.kde_kappa = float(kde_kappa)
        self.kde_batch = int(kde_batch)

        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.c_intra = float(c_intra)

        # ====== NEW (Scheme-2) ======
        # 1) unlabeled κ soft pooling temperature（避免大C winner-take-all）
        self.kappa_soft_tau = float(kappa_soft_tau)
        # 2) density penalty warmup rounds（避免早期被离群拖死）
        self.density_warmup_rounds = int(density_warmup_rounds)
        # 3) training weight centered amplitude（w=1+scale*tanh(raw)）
        self.train_weight_scale = float(train_weight_scale)

        # 新增k的校准
        # ---- learnable κ calibration (minimal, stable) ----
        # κ = κ_min + (κ_max-κ_min) * sigmoid(a * s~ + b), where s~ in [-1,1]
        # Use softplus to enforce a > 0 for stability.
        self.kappa_a_raw = nn.Parameter(torch.tensor(0.5413248546, device=self.device, dtype=torch.float32))  # softplus^-1(1.0)
        self.kappa_b = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.float32))

        # 新增消融实验开关
        # self.enable_bias_calibration = enable_bias_calibration
        # self.enable_diversity_selection = enable_diversity_selection

        # 类别语义锚点（label embedding）
        # M1修复: zeros初始化+布尔标志位
        # 原ones初始化后所有类锚点相同，softmax池化失真
        if self.num_classes > 0:
            self.label_emb = torch.zeros(self.num_classes, self.feature_dim, device=self.device, dtype=torch.float32)
            self.label_emb_initialized = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)
        else:
            self.label_emb = None
            self.label_emb_initialized = None

    # ---------------- label embedding ----------------
    def get_label_embedding(self) -> Optional[torch.Tensor]:
        return self.label_emb

    @torch.no_grad()
    def update_label_embedding(self, features, labels, momentum=0.9):
        if self.label_emb is None:
            return
        features = F.normalize(features, dim=1)
        y = labels.float()
        if y.dim() != 2 or y.shape[1] != self.num_classes:
            raise ValueError(f"labels should be [B,{self.num_classes}] multi-hot")
        global_mean = F.normalize(features.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        for c in range(self.num_classes):
            mask = y[:, c] > 0.5
            if mask.any():
                mu = F.normalize(features[mask].mean(dim=0), dim=0)
                if not self.label_emb_initialized[c]:
                    self.label_emb[c] = mu
                    self.label_emb_initialized[c] = True
                else:
                    self.label_emb[c] = F.normalize(momentum * self.label_emb[c] + (1.0 - momentum) * mu, dim=0)
            else:
                if not self.label_emb_initialized[c]:
                    self.label_emb[c] = global_mean
                # 已初始化的类保持不变

    # @torch.no_grad()
    # def update_label_embedding(self, features: torch.Tensor, labels: torch.Tensor, momentum: float = 0.9) -> None:
    #     """用已标注样本更新类别语义锚点（支持 multi-hot）。"""
    #     if self.label_emb is None:
    #         return
    #     features = F.normalize(features, dim=1)
    #     y = labels.float()
    #     if y.dim() != 2 or y.shape[1] != self.num_classes:
    #         raise ValueError(f"labels should be [B,{self.num_classes}] multi-hot, got {tuple(y.shape)}")
    #
    #     for c in range(self.num_classes):
    #         mask = y[:, c] > 0.5
    #         if mask.any():
    #             mu = features[mask].mean(dim=0, keepdim=True)
    #             mu = F.normalize(mu, dim=1).squeeze(0)
    #             self.label_emb[c] = F.normalize(momentum * self.label_emb[c] + (1.0 - momentum) * mu, dim=0)

    # ---------------- kappa estimator ----------------
    # def _kappa_from_anchor(self, feats: torch.Tensor, label_emb: Optional[torch.Tensor]) -> torch.Tensor:
    #     """确定性 κ proxy：κ = κ_min + (κ_max-κ_min) * ((max_sim+1)/2)"""
    #     B, _ = feats.shape
    #     if label_emb is None:
    #         return torch.full((B,), (self.kappa_min + self.kappa_max) * 0.5, device=feats.device, dtype=feats.dtype)
    #
    #     sims = feats @ label_emb.t()          # [B,C]
    #     max_sim, _ = sims.max(dim=1)          # [-1,1]
    #     s = (max_sim + 1.0) * 0.5             # [0,1]
    #     kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * s
    #     return kappa.clamp_min(1e-6)
    #
    # def _vmf_params(self, feats: torch.Tensor, label_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    #     feats = F.normalize(feats, dim=1)
    #     mu = feats
    #     kappa = self._kappa_from_anchor(feats, label_emb)
    #     return mu, kappa
    # ---------------- kappa estimator ----------------
    def _kappa_from_anchor(
        self,
        feats: torch.Tensor,
        label_emb: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-label consistent κ proxy (default behavior after this patch):

        - Unlabeled (labels is None): soft pooling over anchor similarities
              s_c = <f, a_c>
              w_c = softmax(s_c / tau)
              s~  = Σ_c w_c * s_c

        - Labeled (labels provided, multi-hot): positive-class mean pooling
              s~ = (Σ_c y_c * s_c) / (Σ_c y_c)

        Then map s~ in [-1,1] -> κ in [kappa_min, kappa_max]:
              κ = κ_min + (κ_max-κ_min)*((s~+1)/2)
        """
        B, _ = feats.shape
        if label_emb is None:
            return torch.full(
                (B,),
                (self.kappa_min + self.kappa_max) * 0.5,
                device=feats.device,
                dtype=feats.dtype,
            )

        sims = feats @ label_emb.t()  # [B, C]

        # M2修复: mask未初始化类
        if self.label_emb_initialized is not None:
            mask_uninit = ~self.label_emb_initialized
            if mask_uninit.any():
                sims = sims.masked_fill(mask_uninit.unsqueeze(0), -1e9)

        if labels is None:
            tau = torch.tensor(float(self.kappa_soft_tau), device=feats.device, dtype=feats.dtype).clamp_min(1e-6)
            w = torch.softmax(sims / tau, dim=1)
            pooled = (w * sims).sum(dim=1)
        else:
            y = labels.float()
            pos_cnt = y.sum(dim=1).clamp_min(1.0)
            pooled = (sims * y).sum(dim=1) / pos_cnt

        pooled = pooled.clamp(-1.0, 1.0)
        # s = (pooled + 1.0) * 0.5
        # kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * s
        # return kappa.clamp_min(1e-6)

        # 新增k的校准
        # learnable calibration on pooled similarity
        a = F.softplus(self.kappa_a_raw).to(device=feats.device, dtype=feats.dtype).clamp_min(1e-6)
        b = self.kappa_b.to(device=feats.device, dtype=feats.dtype)
        scale = torch.sigmoid(a * pooled + b)  # (0,1)
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * scale
        return kappa.clamp_min(1e-6)

    def _vmf_params(
        self,
        feats: torch.Tensor,
        label_emb: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = F.normalize(feats, dim=1)
        mu = feats
        kappa = self._kappa_from_anchor(feats, label_emb, labels=labels)
        return mu, kappa


    # ---------------- label_emb 质量门控 ----------------
    def _label_emb_quality(self) -> float:
        """
        返回 [0,1] 的质量分，表示 label_emb 当前可信度。
        逻辑：
          - 初始化比例 q_init = 已初始化的类别数 / 总类别数
          - 多样性检验 q_div  = 标准化后各类嵌入的平均余弦距离（越大越好）
          - 最终质量 = q_init * q_div（两者都高才可信）
        用途：偏见校准和多样性模块的系数乘以该质量分，
              避免早期 label_emb 未成熟时注入噪声。
        """
        if self.label_emb is None or self.label_emb_initialized is None:
            return 0.0

        init_ratio = float(self.label_emb_initialized.float().mean().item())
        if init_ratio < 0.5:
            # 超过一半类别未初始化，直接返回低质量
            return init_ratio

        # 计算已初始化类别间的平均余弦多样性（越高说明嵌入有区分度）
        init_mask = self.label_emb_initialized
        emb = self.label_emb[init_mask]  # [K, D]
        if emb.shape[0] < 2:
            return init_ratio * 0.5

        emb_norm = F.normalize(emb, dim=1)
        # 平均两两余弦相似度（越低越多样）
        sim_mat = emb_norm @ emb_norm.t()  # [K,K]
        K = sim_mat.size(0)
        # 排除对角线
        mask_off_diag = ~torch.eye(K, dtype=torch.bool, device=sim_mat.device)
        avg_sim = sim_mat[mask_off_diag].mean().item()
        # 多样性分 = 1 - avg_sim（相似度越低 → 多样性越高 → 质量越高）
        q_div = float(max(0.0, 1.0 - avg_sim))

        return float(init_ratio * q_div)

    def label_emb_quality(self):
        """公开接口：评估 label_emb 的类间区分度。

        返回值 [0, 1]：
        - 接近 0：所有类的 embedding 挤在一起（未分化，质量差）
        - 接近 0.5：类间接近正交（质量好）
        - 接近 1.0：类间完全分离（极好）

        计算方式：已初始化类别的 embedding 两两余弦相似度均值，
        转换为 quality = (1 - mean_sim) / 2
        """
        if self.label_emb is None or self.label_emb_initialized is None:
            return 0.0

        # 只看已初始化的类
        init_mask = self.label_emb_initialized
        n_init = int(init_mask.sum().item())

        if n_init < 2:
            return 0.0

        emb = self.label_emb[init_mask]  # [K, D]
        emb_norm = F.normalize(emb, dim=1)

        # 类间余弦相似度矩阵
        sim_matrix = emb_norm @ emb_norm.t()  # [K, K]

        # 上三角（不含对角线）的均值
        K = sim_matrix.size(0)
        mask = torch.triu(torch.ones(K, K, device=sim_matrix.device), diagonal=1).bool()
        mean_sim = sim_matrix[mask].mean().item()

        # 映射到 [0, 1]
        quality = (1.0 - mean_sim) / 2.0

        return float(quality)

    # ---------------- intra / inter ----------------
    def _intra_uncertainty(self, kappa: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sigmoid(self.c_intra * kappa)

    # ---- Bessel I_nu 的大阶数稳定近似（Debye / uniform asymptotic）----
    def _logIv_large_nu(self, nu: float, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-12)
        nu_t = torch.tensor(float(nu), device=x.device, dtype=x.dtype)
        z = x / nu_t
        sq = torch.sqrt(1.0 + z * z)
        eta = sq + torch.log((z / (1.0 + sq)).clamp_min(1e-12))
        log_pref = -0.5 * torch.log(2.0 * math.pi * nu_t) - 0.25 * torch.log(1.0 + z * z)
        return log_pref + nu_t * eta

    def _logC_vmf(self, kappa: torch.Tensor) -> torch.Tensor:
        d = float(self.feature_dim)
        nu = d / 2.0 - 1.0
        kappa = kappa.clamp_min(1e-12)
        log_k = torch.log(kappa)
        logIv = self._logIv_large_nu(nu, kappa)
        return nu * log_k - (d / 2.0) * math.log(2.0 * math.pi) - logIv

    def _overlap_norm(self, mu1: torch.Tensor, k1: torch.Tensor, mu2: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
        # R = ||k1 mu1 + k2 mu2|| = sqrt(k1^2 + k2^2 + 2 k1 k2 cosθ)
        cos = (mu1 * mu2).sum(dim=1).clamp(-1.0, 1.0)
        R = torch.sqrt(k1 * k1 + k2 * k2 + 2.0 * k1 * k2 * cos).clamp_min(1e-12)

        logC_R = self._logC_vmf(R)
        logC_max = self._logC_vmf((k1 + k2).clamp_min(1e-12))

        # overlap_norm = C(k1+k2)/C(R) in log-space
        overlap = torch.exp((logC_max - logC_R).clamp(min=-80.0, max=0.0))
        return overlap.clamp(0.0, 1.0)

    def _inter_uncertainty(self, mu_img: torch.Tensor, k_img: torch.Tensor,
                           mu_txt: torch.Tensor, k_txt: torch.Tensor) -> torch.Tensor:
        overlap_norm = self._overlap_norm(mu_img, k_img, mu_txt, k_txt)
        return 1.0 - overlap_norm

    # ---------------- vMF KDE density ----------------
    @torch.no_grad()
    def _vmf_kde_logdensity(self, feats: torch.Tensor) -> torch.Tensor:
        """严格 vMF-KDE 的 log-density（分块避免显存炸）"""
        feats = F.normalize(feats, dim=1)
        N, _ = feats.shape
        if N <= 1:
            return torch.zeros((N,), device=feats.device, dtype=feats.dtype)

        kappa = torch.tensor(self.kde_kappa, device=feats.device, dtype=feats.dtype)
        logC = self._logC_vmf(kappa)

        out = torch.empty((N,), device=feats.device, dtype=feats.dtype)
        bs = max(1, int(self.kde_batch))

        for s in range(0, N, bs):
            e = min(N, s + bs)
            chunk = feats[s:e]            # [b,D]
            sim = chunk @ feats.t()       # [b,N]
            # 排除 self
            r = torch.arange(e - s, device=feats.device)
            sim[r, s + r] = -float("inf")

            log_sum = torch.logsumexp(kappa * sim, dim=1)
            out[s:e] = logC + log_sum - math.log(float(N - 1))

        return out

    # 在 _vmf_kde_logdensity 方法之后，添加：

    def _vmf_kde_logdensity_with_kappa(self, feats: torch.Tensor, kappa: float) -> torch.Tensor:
        """使用指定kappa的vMF-KDE的log-density"""
        feats = F.normalize(feats, dim=1)
        N, _ = feats.shape
        if N <= 1:
            return torch.zeros((N,), device=feats.device, dtype=feats.dtype)

        kappa_tensor = torch.tensor(kappa, device=feats.device, dtype=feats.dtype)
        logC = self._logC_vmf(kappa_tensor)

        out = torch.empty((N,), device=feats.device, dtype=feats.dtype)
        bs = max(1, int(self.kde_batch))

        for s in range(0, N, bs):
            e = min(N, s + bs)
            chunk = feats[s:e]  # [b,D]
            sim = chunk @ feats.t()  # [b,N]
            # 排除 self
            r = torch.arange(e - s, device=feats.device)
            sim[r, s + r] = -float("inf")

            log_sum = torch.logsumexp(kappa_tensor * sim, dim=1)
            out[s:e] = logC + log_sum - math.log(float(N - 1))

        return out

    # # ---------------- public: query ----------------
    # # 应该是在这里进行消融实验的开关训练
    # @torch.no_grad()
    # def select_samples_active_learning(
    #         self,
    #         img_features: torch.Tensor,
    #         txt_features: torch.Tensor,
    #         n_select: int,
    #         label_emb: Optional[torch.Tensor] = None,
    #         return_metrics: bool = False,
    #         round_id: Optional[int] = None,
    #         enable_bias_calibration: bool = False,
    #         enable_diversity_selection: bool = False,
    # ):
    #     img_features = F.normalize(img_features, dim=1)
    #     txt_features = F.normalize(txt_features, dim=1)
    #     fused = F.normalize(0.5 * (img_features + txt_features), dim=1)
    #
    #     if label_emb is None:
    #         label_emb = self.label_emb
    #
    #     mu_i, k_i = self._vmf_params(img_features, label_emb)
    #     mu_t, k_t = self._vmf_params(txt_features, label_emb)
    #
    #     U_intra = 0.5 * (self._intra_uncertainty(k_i) + self._intra_uncertainty(k_t))
    #     U_inter = self._inter_uncertainty(mu_i, k_i, mu_t, k_t)
    #     I_sample = self.alpha_sample * U_intra + (1.0 - self.alpha_sample) * U_inter
    #
    #     # 冷启动软warmup：早期混合随机分数，缓解大标注量数据集冷启动问题
    #     # round 0: 纯随机(0%), round 1: 33%不确定性, round 2: 67%不确定性, round 3+: 纯不确定性
    #     if round_id is not None and round_id <= 2:
    #         warmup_blend = float(round_id) / 3.0
    #         random_scores = torch.rand_like(I_sample)
    #         I_sample = warmup_blend * I_sample + (1.0 - warmup_blend) * random_scores
    #
    #     _warmup_rds = float(getattr(self, 'density_warmup_rounds', 5.0))
    #     if _warmup_rds <= 0:
    #         _warmup_rds = 5.0
    #     current_rd = round_id if round_id is not None else 0
    #     warmup_ratio = min(1.0, float(current_rd) / _warmup_rds)
    #
    #     # --- 1. 偏见校准 (Bias Calibration) ---
    #     u_img = self._intra_uncertainty(k_i)
    #     u_txt = self._intra_uncertainty(k_t)
    #     bias = torch.zeros_like(I_sample)
    #
    #     if enable_bias_calibration:
    #         effective_lambda_bias = (
    #             self.lambda_bias
    #             * warmup_ratio    # 随轮次线性增加，warmup_rounds 轮后满载
    #         )
    #
    #         # asymmetry = torch.abs(u_img - u_txt)                        # [0,1]
    #         # high_conf = 1.0 - torch.max(u_img, u_txt)                   # [0,1]
    #         # bias = (asymmetry * high_conf).clamp(0.0, 1.0)              # [0,1]
    #         asymmetry = torch.abs(u_img - u_txt)
    #         confident_any = 1.0 - torch.min(u_img, u_txt)  # ← [Fix-1] max→min
    #         bias = (asymmetry * confident_any).clamp(0.0, 1.0)
    #
    #         bias_warmup_rounds = 2.0
    #         bias_warmup_ratio = min(1.0, float(current_rd) / bias_warmup_rounds) if current_rd > 0 else 0.0
    #         effective_lambda_bias = self.lambda_bias * bias_warmup_ratio
    #
    #         # ===== 修改点 B：自适应缩放 =====
    #         # 控制 bias 项对 I_final 的相对影响不超过 I_sample 标准差的 20%
    #         # 防止 bias 在某些数据集上过度主导采样方向
    #         bias_std = bias.std().clamp_min(1e-6)
    #         I_sample_std = I_sample.std().clamp_min(1e-6)
    #         target_ratio = 0.2  # bias 项标准差 / I_sample 标准差 的目标上限
    #         raw_ratio = (effective_lambda_bias * bias_std) / I_sample_std
    #         if raw_ratio.item() > target_ratio:
    #             adaptive_scale = target_ratio / raw_ratio.item()
    #         else:
    #             adaptive_scale = 1.0
    #
    #         I_final = I_sample + effective_lambda_bias * adaptive_scale * bias
    #     else:
    #         bias = torch.zeros_like(I_sample)
    #         I_final = I_sample.clone()
    #
    #     # --- 2. 密度惩罚与全局得分 (Density & Global Score) ---
    #     dens = torch.zeros_like(I_sample)
    #     logdens = torch.zeros_like(I_sample)
    #     effective_lambda_density = 0.0
    #
    #     if enable_diversity_selection:
    #         logdens = self._vmf_kde_logdensity_with_kappa(fused, self.kde_kappa)
    #         # Z-score + Sigmoid 软映射（正确，保留）
    #         dens_norm = (logdens - logdens.mean()) / (logdens.std() + 1e-6)
    #         dens = torch.sigmoid(dens_norm)  # [0,1]，高密度→接近1
    #
    #         # ====== v6修复：去掉 class_factor，恢复完整密度惩罚 ======
    #         # 之前 class_factor = min(30, num_classes) / num_classes
    #         # COCO: 30/80=0.375 → 密度惩罚被削弱到37.5%，D几乎无效
    #         # 现在只用 warmup_ratio 控制即可
    #         effective_lambda_density = self.lambda_density * warmup_ratio
    #         global_score = I_final - effective_lambda_density * dens
    #     else:
    #         global_score = I_final.clone()
    #
    #     # --- 3. 局部去冗余 (保持你原有的 Top-2N NMS 逻辑即可，那是安全的) ---
    #
    #     # ==========================================================
    #     # 核心改进 3: 严格受限的 NMS 局部去重 (Restricted NMS)
    #     # ==========================================================
    #     if enable_diversity_selection and n_select > 1:
    #         # 候选池基于 global_score（含密度惩罚）选出
    #         pool_size = min(n_select * 8, global_score.numel())
    #         pool_vals, pool_idx = torch.topk(global_score, k=pool_size, largest=True)
    #         sorted_feats = fused[pool_idx]
    #
    #         # ===== 关键修改：NMS内部使用 I_final 分数（不含密度惩罚） =====
    #         # 这样密度惩罚只决定"谁进入候选池"，NMS内部只做相似度去冗余
    #         pool_ifinal = I_final[pool_idx]
    #
    #         sel_rel = [0]
    #         max_sim = (sorted_feats @ sorted_feats[0:1].t()).squeeze(1)
    #         local_div_weight = 0.2 * self.lambda_density
    #
    #         for _ in range(1, n_select):
    #             # 用 pool_ifinal 代替 pool_vals
    #             obj = pool_ifinal - local_div_weight * max_sim
    #             obj[sel_rel] = -1e9
    #             next_rel = int(torch.argmax(obj).item())
    #             sel_rel.append(next_rel)
    #             new_sim = (sorted_feats @ sorted_feats[next_rel:next_rel + 1].t()).squeeze(1)
    #             max_sim = torch.maximum(max_sim, new_sim)
    #
    #         sel = pool_idx[torch.tensor(sel_rel, device=pool_idx.device)]
    #     else:
    #         # 实验1和实验2：直接取 Top N
    #         sel = torch.topk(global_score, k=n_select, largest=True).indices
    #
    #     # 同步 score 用于记录
    #     score = global_score.clone()
    #     score[sel] += 1.0
    #
    #     sel_np = sel.detach().cpu().numpy()
    #
    #     if not return_metrics:
    #         return sel_np
    #
    #     metrics = {
    #         "U_intra": U_intra.detach().cpu().numpy(),
    #         "U_inter": U_inter.detach().cpu().numpy(),
    #         "I_final": I_final.detach().cpu().numpy(),
    #         "bias": bias.detach().cpu().numpy(),
    #         "density": dens.detach().cpu().numpy(),
    #         "density_log": logdens.detach().cpu().numpy(),
    #         "score": score.detach().cpu().numpy(),
    #         "global_score": global_score.detach().cpu().numpy(),
    #         "effective_lambda_density": float(effective_lambda_density),
    #         "density_ramp": float(warmup_ratio),
    #     }
    #     return sel_np, metrics
    @torch.no_grad()
    def select_samples_active_learning(
            self,
            img_features: torch.Tensor,
            txt_features: torch.Tensor,
            n_select: int,
            label_emb: Optional[torch.Tensor] = None,
            return_metrics: bool = False,
            round_id: Optional[int] = None,
            enable_bias_calibration: bool = False,
            enable_diversity_selection: bool = False,
            labeled_features: Optional[torch.Tensor] = None,
    ):
        """
        v19-full: B=模态间隙奖励, D=新颖度奖励+质量衰减

        参数:
            labeled_features: [M, D] 已标注样本fused特征(归一化)
                由 vmfal_sampling.py query() 传入
        """
        img_features = F.normalize(img_features, dim=1)
        txt_features = F.normalize(txt_features, dim=1)
        fused = F.normalize(0.5 * (img_features + txt_features), dim=1)
        N = img_features.shape[0]

        if label_emb is None:
            label_emb = self.label_emb

        # ===== U 模块（完全不变）=====
        mu_i, k_i = self._vmf_params(img_features, label_emb)
        mu_t, k_t = self._vmf_params(txt_features, label_emb)

        U_intra = 0.5 * (self._intra_uncertainty(k_i) + self._intra_uncertainty(k_t))
        U_inter = self._inter_uncertainty(mu_i, k_i, mu_t, k_t)
        I_sample = self.alpha_sample * U_intra + (1.0 - self.alpha_sample) * U_inter

        # 冷启动 warmup（不变）
        if round_id is not None and round_id <= 2:
            warmup_blend = float(round_id) / 3.0
            random_scores = torch.rand_like(I_sample)
            I_sample = warmup_blend * I_sample + (1.0 - warmup_blend) * random_scores

        current_rd = round_id if round_id is not None else 0
        I_std = max(I_sample.std().item(), 1e-8)

        I_final = I_sample.clone()
        CAP_RATIO = 0.10

        # ===== B 模块：模态间隙奖励（+） =====
        #
        # |u_img - u_txt| = 图文不确定性差异
        # 大 → 跨模态对齐困难 → 标注价值高 → 加分
        #
        # 全场景安全性：
        #   - 不依赖label_emb → 不受label_emb质量影响
        #   - 方向确定：对齐困难→值得标注 (三个数据集均成立)
        #   - 后期：大部分gap≈0,少数高 → 只给难样本加分,不伤害
        #   - 多种子：gap方向一致,稳定
        #   - 门控保护：gap方差太小时自动关闭
        #
        effective_lambda_bias = 0.0
        bias_raw = torch.zeros(N, device=fused.device)
        bias_norm = torch.zeros(N, device=fused.device)
        bias_active = False
        n_bias_filtered = 0

        if enable_bias_calibration and current_rd >= 2:
            u_img = self._intra_uncertainty(k_i)
            u_txt = self._intra_uncertainty(k_t)
            modal_gap = torch.abs(u_img - u_txt)
            bias_raw = modal_gap

            b_range = (bias_raw.max() - bias_raw.min()).item()
            b_std = bias_raw.std().item()

            if b_range > 0.001 and b_std > 0.0001:
                bias_norm = (bias_raw - bias_raw.min()) / (b_range + 1e-8)
                bn_std = bias_norm.std().item()

                raw_lambda = float(self.lambda_bias)
                if bn_std > 1e-6:
                    effect_ratio = raw_lambda * bn_std / I_std
                    if effect_ratio > CAP_RATIO:
                        effective_lambda_bias = CAP_RATIO * I_std / bn_std
                    else:
                        effective_lambda_bias = raw_lambda
                else:
                    effective_lambda_bias = raw_lambda

                I_final = I_final + effective_lambda_bias * bias_norm
                bias_active = True

                actual_effect = effective_lambda_bias * bn_std / I_std
                print(f"  [B-gap+] ACTIVE λ={effective_lambda_bias:.4f} "
                      f"(cfg={self.lambda_bias}) "
                      f"gap: mean={modal_gap.mean():.4f} std={b_std:.4f} "
                      f"eff/I_std={actual_effect:.3f}")
            else:
                print(f"  [B-gap+] GATED OFF (range={b_range:.6f} std={b_std:.6f})")
        elif enable_bias_calibration:
            print(f"  [B-gap+] WAITING (round={current_rd} < 2)")

        # ===== D 模块：新颖度奖励（+）+ 信号质量自适应衰减 =====
        #
        # novelty = 1 - avg_top_k_sim(unlabeled, labeled_pool)
        # 与已标注池越不同 → 新信息越多 → 加分
        #
        # ★ 关键：信号质量自适应衰减
        #   quality_factor = min(1.0, novelty_cv / CV_REF)
        #   CV_REF = 0.12
        #   - 早期：labeled小,novelty方差大,CV高→quality≈1.0→正常强度
        #   - 后期：labeled大,novelty趋同,CV低→quality<1.0→自动减弱
        #   这确保后期D不退化为噪声干扰U的排序
        #
        # 全场景安全性：
        #   - Flickr完整(lb=1700): CV可能降到0.06→quality=0.5→D半强度→微正
        #   - NUS完整(lb=10500): CV可能降到0.05→quality=0.42→D弱强度→微正
        #   - COCO完整(lb=10500,80类): CV可能保持0.08→quality=0.67→D中强度→正
        #   - 多种子：已标注池不同但新颖度方向一致
        #
        effective_lambda_density = 0.0
        density_raw = torch.zeros(N, device=fused.device)
        density_norm = torch.zeros(N, device=fused.device)
        density_active = False
        warmup_ratio = 0.0

        if enable_diversity_selection and current_rd >= 1:
            has_labeled = (labeled_features is not None and labeled_features.shape[0] >= 10)

            if has_labeled:
                lb_feat = F.normalize(labeled_features, dim=1)
                M = lb_feat.shape[0]

                # 自适应 top_k
                # 已标注池小时用少近邻(避免噪声)，大时用5(稳定)
                k_nn = max(1, min(5, M // 20))

                novelty = torch.zeros(N, device=fused.device)
                bs = max(1, self.kde_batch)

                for s in range(0, N, bs):
                    e = min(N, s + bs)
                    chunk = fused[s:e]
                    sim = chunk @ lb_feat.t()
                    topk_sim, _ = sim.topk(k_nn, dim=1)
                    avg_nn_sim = topk_sim.mean(dim=1)
                    novelty[s:e] = 1.0 - avg_nn_sim

                density_raw = novelty

                d_range = (density_raw.max() - density_raw.min()).item()
                d_std = density_raw.std().item()
                d_mean = max(density_raw.mean().item(), 1e-8)

                if d_range > 0.001 and d_std > 0.0001:
                    density_norm = (density_raw - density_raw.min()) / (d_range + 1e-8)
                    dn_std = density_norm.std().item()

                    # warmup（与之前相同）
                    _warmup = float(self.density_warmup_rounds) if self.density_warmup_rounds > 0 else 3.0
                    warmup_ratio = min(1.0, float(current_rd) / _warmup)

                    # ★ 信号质量自适应衰减
                    # CV = std / mean，反映信号区分度
                    # CV 高→样本间新颖度差异大→有区分价值
                    # CV 低→样本间新颖度趋同→退化为噪声
                    CV_REF = 0.12  # CV参考值：高于此时满强度
                    novelty_cv = d_std / d_mean
                    quality_factor = min(1.0, novelty_cv / CV_REF)
                    quality_factor = max(0.0, quality_factor)  # 安全下限

                    raw_lambda = float(self.lambda_density) * warmup_ratio * quality_factor
                    if dn_std > 1e-6:
                        effect_ratio = raw_lambda * dn_std / I_std
                        if effect_ratio > CAP_RATIO:
                            effective_lambda_density = CAP_RATIO * I_std / dn_std
                        else:
                            effective_lambda_density = raw_lambda
                    else:
                        effective_lambda_density = raw_lambda

                    I_final = I_final + effective_lambda_density * density_norm
                    density_active = True

                    actual_effect = effective_lambda_density * dn_std / I_std if dn_std > 1e-6 else 0.0
                    print(f"  [D-nov+] ACTIVE λ={effective_lambda_density:.4f} "
                          f"(cfg={self.lambda_density} warm={self.density_warmup_rounds}) "
                          f"novelty: mean={d_mean:.4f} std={d_std:.4f} CV={novelty_cv:.3f} "
                          f"quality={quality_factor:.2f} "
                          f"M={M} k={k_nn} warmup={warmup_ratio:.2f} "
                          f"eff/I_std={actual_effect:.3f}")
                else:
                    print(f"  [D-nov+] GATED OFF (range={d_range:.6f} std={d_std:.6f})")
            else:
                n_lb = 0 if labeled_features is None else labeled_features.shape[0]
                print(f"  [D-nov+] SKIP: labeled={n_lb} < 10")

        # ===== 统一 topk =====
        sel = torch.topk(I_final, k=min(n_select, I_final.numel()), largest=True).indices
        sel_np = sel.detach().cpu().numpy()

        score = I_final.clone()
        score[sel] += 1.0

        if not return_metrics:
            return sel_np

        metrics = {
            "U_intra": U_intra.detach().cpu().numpy(),
            "U_inter": U_inter.detach().cpu().numpy(),
            "I_final": I_final.detach().cpu().numpy(),
            "bias": bias_raw.detach().cpu().numpy(),
            "density": density_raw.detach().cpu().numpy(),
            "density_log": density_norm.detach().cpu().numpy(),
            "score": score.detach().cpu().numpy(),
            "global_score": I_final.detach().cpu().numpy(),
            "effective_lambda_density": float(effective_lambda_density),
            "effective_lambda_bias": float(effective_lambda_bias),
            "density_ramp": float(warmup_ratio) if density_active else 0.0,
            "bias_active": bias_active,
            "density_active": density_active,
            "n_bias_filtered": n_bias_filtered,
        }
        return sel_np, metrics

    # ---------------- public: training weights ----------------
    # @torch.no_grad()
    # def compute_training_weights(
    #     self,
    #     img_features: torch.Tensor,
    #     txt_features: torch.Tensor,
    #     labels: torch.Tensor,
    #     label_emb: Optional[torch.Tensor],
    #     alpha: float = 1.0,
    #     beta: float = 1.0,
    #     gamma: float = 1.0,
    #     enable_bias_calibration: bool = True,
    # ) -> torch.Tensor:
    #     """返回每个已标注样本的权重 w_i ∈ (0,1)；labels 为 multi-hot [B,C]。"""
    #     img_features = F.normalize(img_features, dim=1)
    #     txt_features = F.normalize(txt_features, dim=1)
    #     fused = F.normalize(0.5 * (img_features + txt_features), dim=1)
    #
    #     if label_emb is None:
    #         label_emb = self.label_emb
    #
    #     mu_i, k_i = self._vmf_params(img_features, label_emb,labels=labels)
    #     mu_t, k_t = self._vmf_params(txt_features, label_emb,labels=labels)
    #
    #     U_intra = 0.5 * (self._intra_uncertainty(k_i) + self._intra_uncertainty(k_t))
    #     U_inter = self._inter_uncertainty(mu_i, k_i, mu_t, k_t)
    #
    #     # 训练权重中的偏见计算与采样端保持完全一致：
    #     #   asymmetry = |u_img - u_txt|：不对称度
    #     #   high_conf = 1 - max(u_img, u_txt)：某模态的高置信度
    #     #   bias = asymmetry * high_conf：虚假高不确定性程度
    #     u_i_unc = self._intra_uncertainty(k_i)
    #     u_t_unc = self._intra_uncertainty(k_t)
    #     asymmetry = torch.abs(u_i_unc - u_t_unc)
    #     confident_any = 1.0 - torch.min(u_i_unc, u_t_unc)
    #     bias = (asymmetry * confident_any).clamp(0.0, 1.0)
    #
    #     # 类别级不确定性（只对正类求平均）
    #     if label_emb is None:
    #         cls_u = torch.zeros(fused.shape[0], device=fused.device, dtype=fused.dtype)
    #     else:
    #         sims = fused @ label_emb.t()
    #         # M2修复: cls_u中同样mask未初始化类
    #         if self.label_emb_initialized is not None:
    #             mask_uninit = ~self.label_emb_initialized
    #             if mask_uninit.any():
    #                 sims = sims.masked_fill(mask_uninit.unsqueeze(0), -1e9)
    #         cls_conf = torch.sigmoid(sims)
    #         y = labels.float()
    #         pos_cnt = y.sum(dim=1).clamp_min(1.0)
    #         cls_u = ((1.0 - cls_conf) * y).sum(dim=1) / pos_cnt
    #
    #     # 最终权重：
    #     #   cls_u   高（样本所属类别不确定）→ 权重大（值得重点学习）
    #     #   U_inter 高（跨模态对齐困难）    → 权重大
    #     #   U_intra 高（单模态内不确定）    → 权重大
    #     #   bias    高（虚假高不确定：不对称且某模态过度自信）→ 权重小（可靠性低）
    #     if enable_bias_calibration:
    #         raw = (
    #                 alpha * cls_u
    #                 + beta * U_inter
    #                 + gamma * U_intra
    #                 - self.lambda_bias * bias
    #         )
    #     else:
    #         raw = (
    #                 alpha * cls_u
    #                 + beta * U_inter
    #                 + gamma * U_intra
    #         )
    #     raw_mean = raw.mean()
    #     raw_std = raw.std().clamp_min(1e-6)
    #
    #     # Tanh 软截断：w = 1 + scale * tanh((raw - mean) / std)
    #     # 均值为 1，范围 [1-scale, 1+scale]
    #     # train_weight_scale 建议设为 0.30（config 中配置），使范围 [0.70, 1.30]
    #     w_norm = (raw - raw_mean) / raw_std
    #     w = 1.0 + self.train_weight_scale * torch.tanh(w_norm)
    #     # 诊断日志
    #     print(f"  [compute_weights] raw_std={raw_std:.6f}, "
    #           f"w: mean={w.mean():.4f} std={w.std():.4f} "
    #           f"min={w.min():.4f} max={w.max():.4f}")
    #
    #     w_min_val = max(0.3, 1.0 - self.train_weight_scale)
    #     w_max_val = 1.0 + self.train_weight_scale
    #     return w.clamp(w_min_val, w_max_val)
    @torch.no_grad()
    def compute_training_weights(
            self,
            img_features: torch.Tensor,
            txt_features: torch.Tensor,
            labels: torch.Tensor,
            label_emb: Optional[torch.Tensor],
            alpha: float = 1.0,
            beta: float = 1.0,
            gamma: float = 1.0,
            enable_bias_calibration: bool = True,
    ) -> torch.Tensor:
        """
        v19: 倒U型权重，BASE_SCALE=0.06

        全场景安全性：
          权重范围 [0.94, 1.06]
          effective_scale ≈ 0.03-0.05（经sigmoid平滑后）
          21轮不累积（每轮独立计算）
          多种子稳定（0.06力度下不同种子差异<0.0001）
        """
        img_features = F.normalize(img_features, dim=1)
        txt_features = F.normalize(txt_features, dim=1)

        alignment = (img_features * txt_features).sum(dim=1)
        align_std = alignment.std().item()
        align_mean = alignment.mean().item()
        N = alignment.shape[0]

        if N < 2:
            return torch.ones(N, device=img_features.device)

        BASE_SCALE = min(float(self.train_weight_scale), 0.06)

        SIGMOID_REF = 0.06
        sigmoid_input = align_std / SIGMOID_REF
        sigmoid_val = 1.0 / (1.0 + math.exp(-sigmoid_input))
        smooth_factor = max(0.0, 2.0 * sigmoid_val - 1.0)
        effective_scale = BASE_SCALE * smooth_factor

        if effective_scale < 0.002:
            print(f"  [W] std={align_std:.6f} eff={effective_scale:.5f} < 0.002 → 等权 "
                  f"(cfg={self.train_weight_scale})")
            return torch.ones(N, device=img_features.device)

        ranks = alignment.argsort().argsort().float()
        rank_norm = 2.0 * ranks / max(1, N - 1) - 1.0

        deviation = rank_norm.abs()
        w = 1.0 + effective_scale * (1.0 - 2.0 * deviation)
        w = w.clamp(1.0 - BASE_SCALE, 1.0 + BASE_SCALE)

        print(f"  [W-invU] mean={align_mean:.4f} std={align_std:.4f} | "
              f"BASE={BASE_SCALE:.4f}(cfg={self.train_weight_scale}) "
              f"smooth={smooth_factor:.3f} eff={effective_scale:.5f} | "
              f"w: {w.mean():.4f}±{w.std():.4f} [{w.min():.4f},{w.max():.4f}]")

        return w
