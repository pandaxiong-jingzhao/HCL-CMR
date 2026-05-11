# models/vmfal_model.py
"""
vMF Uncertainty Model v32

三模块架构：
  U: 基于 vMF 分布的超球面不确定性估计（采样端，核心）
  D: 类别覆盖多样性（采样端，辅助）
  W: 难度自适应训练权重（训练端，辅助）
     - 融合两个正交难度维度：
       (1) 跨模态对齐难度：投影后 gap = 1 - cos(proj_img, proj_txt)
       (2) 语义分类难度：锚点距离 = 1 - cos(fused, anchor)

消融设计：
  Full:  采样 U+D, 训练 W加权
  wo_D:  采样 U,   训练 W加权     → 证明D有效
  wo_W:  采样 U+D, 训练 等权      → 证明W有效

  Full和wo_W使用完全相同的采样结果（U+D），消融只反映加权效果。
  wo_D和Full采样不同，但D已经在之前的实验中稳定通过。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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

        self.kappa_soft_tau = float(kappa_soft_tau)
        self.density_warmup_rounds = int(density_warmup_rounds)
        self.train_weight_scale = float(train_weight_scale)

        self.kappa_a_raw = nn.Parameter(
            torch.tensor(0.5413248546, device=self.device, dtype=torch.float32)
        )
        self.kappa_b = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=torch.float32)
        )

        if self.num_classes > 0:
            self.label_emb = torch.zeros(
                self.num_classes, self.feature_dim,
                device=self.device, dtype=torch.float32
            )
            self.label_emb_initialized = torch.zeros(
                self.num_classes, dtype=torch.bool, device=self.device
            )
        else:
            self.label_emb = None
            self.label_emb_initialized = None

    # -------------------- label embedding --------------------
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
        global_mean = F.normalize(
            features.mean(dim=0, keepdim=True), dim=1
        ).squeeze(0)
        for c in range(self.num_classes):
            mask = y[:, c] > 0.5
            if mask.any():
                mu = F.normalize(features[mask].mean(dim=0), dim=0)
                if not self.label_emb_initialized[c]:
                    self.label_emb[c] = mu
                    self.label_emb_initialized[c] = True
                else:
                    self.label_emb[c] = F.normalize(
                        momentum * self.label_emb[c] + (1.0 - momentum) * mu, dim=0
                    )
            else:
                if not self.label_emb_initialized[c]:
                    self.label_emb[c] = global_mean

    # -------------------- kappa estimator --------------------
    def _kappa_from_anchor(self, feats, label_emb, labels=None):
        B, _ = feats.shape
        if label_emb is None:
            return torch.full(
                (B,), (self.kappa_min + self.kappa_max) * 0.5,
                device=feats.device, dtype=feats.dtype
            )
        sims = feats @ label_emb.t()
        if self.label_emb_initialized is not None:
            mask_uninit = ~self.label_emb_initialized
            if mask_uninit.any():
                sims = sims.masked_fill(mask_uninit.unsqueeze(0), -1e9)
        if labels is None:
            tau = torch.tensor(
                float(self.kappa_soft_tau), device=feats.device, dtype=feats.dtype
            ).clamp_min(1e-6)
            w = torch.softmax(sims / tau, dim=1)
            pooled = (w * sims).sum(dim=1)
        else:
            y = labels.float()
            pos_cnt = y.sum(dim=1).clamp_min(1.0)
            pooled = (sims * y).sum(dim=1) / pos_cnt
        pooled = pooled.clamp(-1.0, 1.0)
        a = F.softplus(self.kappa_a_raw).to(
            device=feats.device, dtype=feats.dtype
        ).clamp_min(1e-6)
        b = self.kappa_b.to(device=feats.device, dtype=feats.dtype)
        scale = torch.sigmoid(a * pooled + b)
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * scale
        return kappa.clamp_min(1e-6)

    def _vmf_params(self, feats, label_emb, labels=None):
        feats = F.normalize(feats, dim=1)
        mu = feats
        kappa = self._kappa_from_anchor(feats, label_emb, labels=labels)
        return mu, kappa

    # -------------------- quality --------------------
    def _label_emb_quality(self) -> float:
        if self.label_emb is None or self.label_emb_initialized is None:
            return 0.0
        init_ratio = float(self.label_emb_initialized.float().mean().item())
        if init_ratio < 0.5:
            return init_ratio
        init_mask = self.label_emb_initialized
        emb = self.label_emb[init_mask]
        if emb.shape[0] < 2:
            return init_ratio * 0.5
        emb_norm = F.normalize(emb, dim=1)
        sim_mat = emb_norm @ emb_norm.t()
        K = sim_mat.size(0)
        mask_off_diag = ~torch.eye(K, dtype=torch.bool, device=sim_mat.device)
        avg_sim = sim_mat[mask_off_diag].mean().item()
        q_div = float(max(0.0, 1.0 - avg_sim))
        return float(init_ratio * q_div)

    def label_emb_quality(self):
        if self.label_emb is None or self.label_emb_initialized is None:
            return 0.0
        init_mask = self.label_emb_initialized
        n_init = int(init_mask.sum().item())
        if n_init < 2:
            return 0.0
        emb = self.label_emb[init_mask]
        emb_norm = F.normalize(emb, dim=1)
        sim_matrix = emb_norm @ emb_norm.t()
        K = sim_matrix.size(0)
        mask = torch.triu(
            torch.ones(K, K, device=sim_matrix.device), diagonal=1
        ).bool()
        mean_sim = sim_matrix[mask].mean().item()
        quality = (1.0 - mean_sim) / 2.0
        return float(quality)

    # -------------------- intra / inter --------------------
    def _intra_uncertainty(self, kappa):
        return 1.0 - torch.sigmoid(self.c_intra * kappa)

    def _logIv_large_nu(self, nu, x):
        x = x.clamp_min(1e-12)
        nu_t = torch.tensor(float(nu), device=x.device, dtype=x.dtype)
        z = x / nu_t
        sq = torch.sqrt(1.0 + z * z)
        eta = sq + torch.log((z / (1.0 + sq)).clamp_min(1e-12))
        log_pref = -0.5 * torch.log(2.0 * math.pi * nu_t) - 0.25 * torch.log(1.0 + z * z)
        return log_pref + nu_t * eta

    def _logC_vmf(self, kappa):
        d = float(self.feature_dim)
        nu = d / 2.0 - 1.0
        kappa = kappa.clamp_min(1e-12)
        log_k = torch.log(kappa)
        logIv = self._logIv_large_nu(nu, kappa)
        return nu * log_k - (d / 2.0) * math.log(2.0 * math.pi) - logIv

    def _overlap_norm(self, mu1, k1, mu2, k2):
        cos = (mu1 * mu2).sum(dim=1).clamp(-1.0, 1.0)
        R = torch.sqrt(k1 * k1 + k2 * k2 + 2.0 * k1 * k2 * cos).clamp_min(1e-12)
        logC_R = self._logC_vmf(R)
        logC_max = self._logC_vmf((k1 + k2).clamp_min(1e-12))
        overlap = torch.exp((logC_max - logC_R).clamp(min=-80.0, max=0.0))
        return overlap.clamp(0.0, 1.0)

    def _inter_uncertainty(self, mu_img, k_img, mu_txt, k_txt):
        overlap_norm = self._overlap_norm(mu_img, k_img, mu_txt, k_txt)
        return 1.0 - overlap_norm

    # -------------------- vMF KDE --------------------
    @torch.no_grad()
    def _vmf_kde_logdensity(self, feats):
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
            chunk = feats[s:e]
            sim = chunk @ feats.t()
            r = torch.arange(e - s, device=feats.device)
            sim[r, s + r] = -float("inf")
            log_sum = torch.logsumexp(kappa * sim, dim=1)
            out[s:e] = logC + log_sum - math.log(float(N - 1))
        return out

    def _vmf_kde_logdensity_with_kappa(self, feats, kappa):
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
            chunk = feats[s:e]
            sim = chunk @ feats.t()
            r = torch.arange(e - s, device=feats.device)
            sim[r, s + r] = -float("inf")
            log_sum = torch.logsumexp(kappa_tensor * sim, dim=1)
            out[s:e] = logC + log_sum - math.log(float(N - 1))
        return out

    # ================================================================
    # select_samples_active_learning（v32：U + D，不含B/W）
    # ================================================================
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
            labeled_labels: Optional[torch.Tensor] = None,
            # 保留旧接口兼容
            projected_gap: Optional[torch.Tensor] = None,
            projected_img: Optional[torch.Tensor] = None,
            projected_txt: Optional[torch.Tensor] = None,
            projected_mask: Optional[torch.Tensor] = None,
    ):
        """
        v32 采样：U + D

        采样端不含B和W。训练端权重在 compute_training_weights 中处理。
        enable_bias_calibration 参数保留但不影响采样。
        """
        img_features = F.normalize(img_features, dim=1)
        txt_features = F.normalize(txt_features, dim=1)
        fused = F.normalize(0.5 * (img_features + txt_features), dim=1)
        N = img_features.shape[0]

        if label_emb is None:
            label_emb = self.label_emb

        # ===== U 模块 =====
        mu_i, k_i = self._vmf_params(img_features, label_emb)
        mu_t, k_t = self._vmf_params(txt_features, label_emb)

        U_intra = 0.5 * (self._intra_uncertainty(k_i) + self._intra_uncertainty(k_t))
        U_inter = self._inter_uncertainty(mu_i, k_i, mu_t, k_t)
        I_sample = self.alpha_sample * U_intra + (1.0 - self.alpha_sample) * U_inter

        # 冷启动 warmup
        if round_id is not None and round_id <= 2:
            warmup_blend = float(round_id) / 3.0
            random_scores = torch.rand_like(I_sample)
            I_sample = warmup_blend * I_sample + (1.0 - warmup_blend) * random_scores

        current_rd = round_id if round_id is not None else 0
        I_std = max(I_sample.std().item(), 1e-8)
        I_final = I_sample.clone()
        CAP_RATIO = 0.12

        # ===== D 模块 =====
        effective_lambda_density = 0.0
        density_raw = torch.zeros(N, device=fused.device)
        density_norm = torch.zeros(N, device=fused.device)
        density_active = False
        warmup_ratio = 0.0

        if enable_diversity_selection and current_rd >= 1:
            has_labeled = (
                labeled_labels is not None and labeled_labels.shape[0] >= 10
            )
            has_label_emb = (
                label_emb is not None
                and self.label_emb_initialized is not None
                and self.label_emb_initialized.sum().item()
                >= max(2, self.num_classes // 2)
            )

            if has_labeled and has_label_emb:
                y_lb = labeled_labels.float()
                M = y_lb.shape[0]
                C = y_lb.shape[1]

                class_count = y_lb.sum(dim=0)
                class_freq = class_count / max(float(M), 1.0)

                MIN_SAMPLES_FOR_WEIGHT = 3
                eps_freq = 1e-4
                class_weight = 1.0 / torch.sqrt(class_freq + eps_freq)

                sparse_mask = class_count < MIN_SAMPLES_FOR_WEIGHT
                if sparse_mask.any():
                    n_sparse = int(sparse_mask.sum().item())
                    print(
                        f"  [D-cls] 长尾保护: {n_sparse} classes "
                        f"<{MIN_SAMPLES_FOR_WEIGHT} samples"
                    )

                if self.label_emb_initialized is not None:
                    uninit_mask = ~self.label_emb_initialized
                    sparse_mask = sparse_mask | uninit_mask

                valid_mask = ~sparse_mask
                if valid_mask.sum() > 0:
                    valid_weights = class_weight[valid_mask]
                    valid_mean = valid_weights.mean().clamp_min(1e-8)
                    class_weight[valid_mask] = valid_weights / valid_mean
                class_weight[sparse_mask] = 1.0
                class_weight = class_weight.clamp(0.2, 5.0)

                le_norm = F.normalize(label_emb, dim=1)
                sim_to_class = fused @ le_norm.t()
                tau_d = max(float(self.kappa_soft_tau), 0.05)
                affinity = torch.softmax(sim_to_class / tau_d, dim=1)
                diversity_score = (
                    affinity * class_weight.unsqueeze(0)
                ).sum(dim=1)
                density_raw = diversity_score

                d_range = (density_raw.max() - density_raw.min()).item()
                d_std = density_raw.std().item()

                if d_range > 0.001 and d_std > 0.0001:
                    density_norm = (density_raw - density_raw.min()) / (
                        d_range + 1e-8
                    )
                    dn_std = density_norm.std().item()
                    _warmup = (
                        float(self.density_warmup_rounds)
                        if self.density_warmup_rounds > 0
                        else 3.0
                    )
                    warmup_ratio = min(1.0, float(current_rd) / _warmup)
                    raw_lambda = float(self.lambda_density) * warmup_ratio
                    if dn_std > 1e-6:
                        effect_ratio = raw_lambda * dn_std / I_std
                        if effect_ratio > CAP_RATIO:
                            effective_lambda_density = (
                                CAP_RATIO * I_std / dn_std
                            )
                        else:
                            effective_lambda_density = raw_lambda
                    else:
                        effective_lambda_density = raw_lambda
                    I_final = I_final + effective_lambda_density * density_norm
                    density_active = True
                    actual_effect = (
                        effective_lambda_density * dn_std / I_std
                        if dn_std > 1e-6
                        else 0.0
                    )
                    print(
                        f"  [D-cls] ACTIVE λ={effective_lambda_density:.4f} "
                        f"(cfg={self.lambda_density} "
                        f"warm={self.density_warmup_rounds}) "
                        f"warmup={warmup_ratio:.2f} "
                        f"eff/I_std={actual_effect:.3f}"
                    )
                else:
                    print(
                        f"  [D-cls] GATED OFF "
                        f"(range={d_range:.6f} std={d_std:.6f})"
                    )
            else:
                n_lb = (
                    0 if labeled_labels is None else labeled_labels.shape[0]
                )
                n_init = (
                    0
                    if self.label_emb_initialized is None
                    else int(self.label_emb_initialized.sum().item())
                )
                print(
                    f"  [D-cls] SKIP: labeled={n_lb}, "
                    f"label_emb_init={n_init}/{self.num_classes}"
                )

        # v32: 采样端不含 B，不含 W
        effective_lambda_bias = 0.0
        bias_norm = torch.zeros(N, device=fused.device)

        # ---- topk ----
        sel = torch.topk(
            I_final, k=min(n_select, I_final.numel()), largest=True
        ).indices
        sel_np = sel.detach().cpu().numpy()

        if not return_metrics:
            return sel_np

        metrics = {
            "U_intra": float(U_intra.mean().item()),
            "U_inter": float(U_inter.mean().item()),
            "I_final": float(I_final.mean().item()),
            "bias": 0.0,
            "density": float(density_raw.mean().item()),
            "density_log": float(density_norm.mean().item()),
            "score": float(I_final.mean().item()),
            "global_score": float(I_final.mean().item()),
            "effective_lambda_density": float(effective_lambda_density),
            "effective_lambda_bias": 0.0,
            "density_ramp": float(warmup_ratio) if density_active else 0.0,
            "bias_active": False,
            "density_active": density_active,
            "n_bias_reranked": 0,
        }
        return sel_np, metrics

    # ================================================================
    # compute_training_weights（v32：双维度难度加权）
    # ================================================================
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
            pan_losses: Optional[torch.Tensor] = None,
            projected_alignment: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        v32: 双维度难度自适应权重

        维度1 - 跨模态对齐难度：
          gap = 1 - cos(proj_img, proj_txt)
          gap大 → 图文方向不一致 → 模型需要更多学习跨模态对齐
          来源：projected_alignment（train()中投影已标注样本时顺便计算）

        维度2 - 语义分类难度：
          dist = 1 - cos(fused, pos_anchor_mean)
          距离大 → 远离类中心 → 类别边界附近的难样本
          来源：label_emb（类别语义锚点）

        两个维度正交：
          可以跨模态一致但远离类中心（维度1小，维度2大）
          可以跨模态不一致但在类中心附近（维度1大，维度2小）

        消融控制：
          enable_bias_calibration=False → 去掉维度1（wo_W消融配置）
          label_emb未就绪 → 自动降级只用维度1
        """
        N = img_features.shape[0]
        if N < 2:
            return torch.ones(N, device=img_features.device)

        img_features = F.normalize(img_features, dim=1)
        txt_features = F.normalize(txt_features, dim=1)
        fused = F.normalize(0.5 * (img_features + txt_features), dim=1)

        SCALE = min(max(float(self.train_weight_scale), 0.05), 0.50)

        # 收集有效的难度信号
        difficulty_signals = []
        signal_names = []

        # ---- 维度1：跨模态对齐难度 ----
        if enable_bias_calibration:
            if projected_alignment is not None and projected_alignment.numel() == N:
                bias_difficulty = 1.0 - projected_alignment.to(img_features.device)
                source = "projected_gap"
            else:
                raw_align = (img_features * txt_features).sum(dim=1)
                bias_difficulty = 1.0 - raw_align
                source = "raw_gap"

            # rank transform
            b_sorted = bias_difficulty.argsort()
            b_ranks = torch.zeros_like(bias_difficulty)
            b_ranks[b_sorted] = torch.linspace(
                0, 1, N, device=bias_difficulty.device
            )
            difficulty_signals.append(b_ranks)
            signal_names.append(source)

            print(
                f"  [W-dim1] {source}: "
                f"mean={bias_difficulty.mean():.4f} "
                f"std={bias_difficulty.std():.4f}"
            )

        # ---- 维度2：语义分类难度 ----
        has_emb = (
            label_emb is not None
            and self.label_emb_initialized is not None
            and self.label_emb_initialized.sum().item() >= 2
        )

        if has_emb:
            init_mask = self.label_emb_initialized
            emb_init = label_emb[init_mask]
            emb_norm = F.normalize(emb_init, dim=1)
            sim_to_class = fused @ emb_norm.t()

            y = labels.float()
            y_init = y[:, init_mask]
            pos_cnt = y_init.sum(dim=1).clamp_min(1.0)
            anchor_proximity = (sim_to_class * y_init).sum(dim=1) / pos_cnt

            no_pos = y_init.sum(dim=1) == 0
            if no_pos.any():
                anchor_proximity[no_pos] = sim_to_class[no_pos].max(
                    dim=1
                ).values

            anchor_difficulty = 1.0 - anchor_proximity.clamp(-1.0, 1.0)

            # rank transform
            a_sorted = anchor_difficulty.argsort()
            a_ranks = torch.zeros_like(anchor_difficulty)
            a_ranks[a_sorted] = torch.linspace(
                0, 1, N, device=anchor_difficulty.device
            )
            difficulty_signals.append(a_ranks)
            signal_names.append("anchor_dist")

            n_init = int(init_mask.sum().item())
            print(
                f"  [W-dim2] anchor_dist: "
                f"proximity mean={anchor_proximity.mean():.4f} "
                f"std={anchor_proximity.std():.4f} "
                f"C_init={n_init}/{self.num_classes}"
            )

        # ---- 合成 ----
        if len(difficulty_signals) == 0:
            print(f"  [W] 无有效信号，返回等权")
            return torch.ones(N, device=img_features.device)

        # 平均各维度的 rank
        combined_rank = torch.stack(difficulty_signals, dim=0).mean(dim=0)

        # sigmoid 映射
        weight_raw = torch.sigmoid(4.0 * (combined_rank - 0.5)) - 0.5
        w = 1.0 + SCALE * weight_raw

        w_min = max(0.5, 1.0 - SCALE * 0.6)
        w_max = min(2.0, 1.0 + SCALE * 0.6)
        w = w.clamp(w_min, w_max)
        w = w / (w.mean() + 1e-8)

        print(
            f"  [W-final] signals=[{','.join(signal_names)}] "
            f"SCALE={SCALE:.3f} | "
            f"w: {w.mean():.4f}±{w.std():.4f} "
            f"[{w.min():.3f},{w.max():.3f}]"
        )

        return w