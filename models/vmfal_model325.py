# models/vmfal_model.py
"""
vMF Uncertainty Model v20

核心模块 U：基于 vMF 分布的超球面不确定性估计
辅助模块 B：跨模态方向差异（超球面测地距离）
辅助模块 D：类别覆盖多样性（基于超球面类别锚点的亲和度加权）
辅助模块 W：难度自适应训练权重（基于 PAN loss）

- inter-uncertainty：用 vMF 分布重叠度闭式（产品积分）并归一化到 (0,1]
    overlap_norm = C_d(κi+κt) / C_d(R),  R = ||κi μi + κt μt||
    U_inter = 1 - overlap_norm
- κ 的估计：使用"与类别锚点相似度"的确定性 proxy
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

        # learnable κ calibration
        self.kappa_a_raw = nn.Parameter(torch.tensor(0.5413248546, device=self.device, dtype=torch.float32))
        self.kappa_b = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.float32))

        # 类别语义锚点（label embedding）
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

    # ---------------- kappa estimator ----------------
    def _kappa_from_anchor(
        self,
        feats: torch.Tensor,
        label_emb: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _ = feats.shape
        if label_emb is None:
            return torch.full(
                (B,),
                (self.kappa_min + self.kappa_max) * 0.5,
                device=feats.device,
                dtype=feats.dtype,
            )

        sims = feats @ label_emb.t()  # [B, C]

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

        a = F.softplus(self.kappa_a_raw).to(device=feats.device, dtype=feats.dtype).clamp_min(1e-6)
        b = self.kappa_b.to(device=feats.device, dtype=feats.dtype)
        scale = torch.sigmoid(a * pooled + b)
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
        mask = torch.triu(torch.ones(K, K, device=sim_matrix.device), diagonal=1).bool()
        mean_sim = sim_matrix[mask].mean().item()
        quality = (1.0 - mean_sim) / 2.0
        return float(quality)

    # ---------------- intra / inter ----------------
    def _intra_uncertainty(self, kappa: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sigmoid(self.c_intra * kappa)

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

    # ---------------- vMF KDE density ----------------
    @torch.no_grad()
    def _vmf_kde_logdensity(self, feats: torch.Tensor) -> torch.Tensor:
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

    def _vmf_kde_logdensity_with_kappa(self, feats: torch.Tensor, kappa: float) -> torch.Tensor:
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
    # 核心方法1：select_samples_active_learning（v20 重写）
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
            projected_img: Optional[torch.Tensor] = None,
            projected_txt: Optional[torch.Tensor] = None,
            projected_mask: Optional[torch.Tensor] = None,
    ):
        """
        v25 采样函数
        U = 超球面vMF不确定性（核心）
        B = 跨模态方向差异（v25：候选内重排序，不对全量加bias）
        D = 类别覆盖多样性（基于超球面类别锚点）
        W = 训练加权（在训练阶段生效，不在采样中）

        v25关键改进：B模块不再对全量N个样本加分，而是在U+D选出候选后，
        在候选内部做direction_gap加权重排序。确保B模块信号100%有效。
        """
        img_features = F.normalize(img_features, dim=1)
        txt_features = F.normalize(txt_features, dim=1)
        fused = F.normalize(0.5 * (img_features + txt_features), dim=1)
        N = img_features.shape[0]

        if label_emb is None:
            label_emb = self.label_emb

        # ===== U 模块（核心）=====
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

        # ===== D 模块（对全量加分，不变）=====
        effective_lambda_density = 0.0
        density_raw = torch.zeros(N, device=fused.device)
        density_norm = torch.zeros(N, device=fused.device)
        density_active = False
        warmup_ratio = 0.0

        if enable_diversity_selection and current_rd >= 1:
            has_labeled = (labeled_labels is not None and labeled_labels.shape[0] >= 10)
            has_label_emb = (label_emb is not None and self.label_emb_initialized is not None
                             and self.label_emb_initialized.sum().item() >= max(2, self.num_classes // 2))

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
                    print(f"  [D-cls] 长尾保护: {n_sparse} classes with <{MIN_SAMPLES_FOR_WEIGHT} samples → weight=1")

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

                diversity_score = (affinity * class_weight.unsqueeze(0)).sum(dim=1)
                density_raw = diversity_score

                d_range = (density_raw.max() - density_raw.min()).item()
                d_std = density_raw.std().item()

                if d_range > 0.001 and d_std > 0.0001:
                    density_norm = (density_raw - density_raw.min()) / (d_range + 1e-8)
                    dn_std = density_norm.std().item()

                    _warmup = float(self.density_warmup_rounds) if self.density_warmup_rounds > 0 else 3.0
                    warmup_ratio = min(1.0, float(current_rd) / _warmup)

                    raw_lambda = float(self.lambda_density) * warmup_ratio
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

                    freq_valid = class_freq[class_freq > eps_freq]
                    imbalance_ratio = (freq_valid.max() / freq_valid.min()).item() if freq_valid.numel() > 1 else 1.0

                    actual_effect = effective_lambda_density * dn_std / I_std if dn_std > 1e-6 else 0.0
                    print(f"  [D-cls] ACTIVE λ={effective_lambda_density:.4f} "
                          f"(cfg={self.lambda_density} warm={self.density_warmup_rounds}) "
                          f"class_imbalance={imbalance_ratio:.1f}x "
                          f"M={M} C_init={int(self.label_emb_initialized.sum().item())}/{C} "
                          f"sparse={int(sparse_mask.sum().item())} "
                          f"warmup={warmup_ratio:.2f} "
                          f"eff/I_std={actual_effect:.3f}")
                else:
                    print(f"  [D-cls] GATED OFF (range={d_range:.6f} std={d_std:.6f})")
            else:
                n_lb = 0 if labeled_labels is None else labeled_labels.shape[0]
                n_init = 0 if self.label_emb_initialized is None else int(self.label_emb_initialized.sum().item())
                print(f"  [D-cls] SKIP: labeled={n_lb}, label_emb_init={n_init}/{self.num_classes}")

        # ===== B 模块（v25：候选内重排序）=====
        # 思路：先用 U+D 选出 2×n_select 个候选，再在候选内用 direction_gap 重排
        # 这样B模块只影响候选内部排序，信号100%有效
        effective_lambda_bias = 0.0
        bias_raw = torch.zeros(N, device=fused.device)
        bias_active = False
        n_bias_reranked = 0

        if enable_bias_calibration and current_rd >= 1:
            # Step 1: U+D 先选出候选池（2倍大小）
            n_candidate = min(2 * n_select, N)
            candidate_indices = torch.topk(I_final, k=n_candidate, largest=True).indices

            # Step 2: 在候选内计算 direction_gap
            has_projection = (projected_img is not None and projected_txt is not None
                              and projected_mask is not None)

            if has_projection:
                # 取候选子集的投影特征
                cand_proj_img = projected_img[candidate_indices]
                cand_proj_txt = projected_txt[candidate_indices]
                cand_mask = projected_mask[candidate_indices]

                # 候选中有投影的部分
                n_proj_in_cand = int(cand_mask.sum().item())

                if n_proj_in_cand >= 10:
                    # 对有投影的位置用投影特征，无投影的用原始特征
                    cand_img_n = F.normalize(cand_proj_img, dim=1)
                    cand_txt_n = F.normalize(cand_proj_txt, dim=1)
                    cand_gap = 1.0 - (cand_img_n * cand_txt_n).sum(dim=1)

                    # 候选内归一化
                    g_min = cand_gap.min()
                    g_max = cand_gap.max()
                    g_range = (g_max - g_min).item()
                    g_std = cand_gap.std().item()

                    if g_range > 0.001 and g_std > 0.0001:
                        cand_gap_norm = (cand_gap - g_min) / (g_range + 1e-8)

                        # 在候选内的 I_final 上加 bias
                        cand_scores = I_final[candidate_indices].clone()
                        cand_I_std = max(cand_scores.std().item(), 1e-8)

                        b_warmup = min(1.0, float(current_rd) / 3.0)
                        raw_lambda = float(self.lambda_bias) * b_warmup
                        gn_std = cand_gap_norm.std().item()

                        if gn_std > 1e-6:
                            effect_ratio = raw_lambda * gn_std / cand_I_std
                            if effect_ratio > CAP_RATIO:
                                effective_lambda_bias = CAP_RATIO * cand_I_std / gn_std
                            else:
                                effective_lambda_bias = raw_lambda
                        else:
                            effective_lambda_bias = raw_lambda

                        cand_scores = cand_scores + effective_lambda_bias * cand_gap_norm

                        # 从重排序后的候选中选 top n_select
                        sel_in_cand = torch.topk(cand_scores, k=min(n_select, n_candidate), largest=True).indices
                        sel = candidate_indices[sel_in_cand]
                        bias_active = True
                        n_bias_reranked = n_candidate

                        actual_effect = effective_lambda_bias * gn_std / cand_I_std
                        print(f"  [B-dir] RERANK active: λ={effective_lambda_bias:.4f} "
                              f"(cfg={self.lambda_bias} warmup={b_warmup:.2f}) "
                              f"candidates={n_candidate} proj_in_cand={n_proj_in_cand} "
                              f"gap: mean={cand_gap.mean():.4f} std={g_std:.4f} "
                              f"range=[{g_min:.4f},{g_max:.4f}] "
                              f"eff/cand_std={actual_effect:.3f}")
                    else:
                        print(f"  [B-dir] GATED OFF in candidates (range={g_range:.6f} std={g_std:.6f})")
                        sel = candidate_indices[:n_select]
                else:
                    print(f"  [B-dir] SKIP: too few projected in candidates ({n_proj_in_cand}/{n_candidate})")
                    sel = candidate_indices[:n_select]
            else:
                # 无投影时：用原始特征在候选内做 rerank
                cand_img = img_features[candidate_indices]
                cand_txt = txt_features[candidate_indices]
                cand_gap = 1.0 - (cand_img * cand_txt).sum(dim=1)

                g_range = (cand_gap.max() - cand_gap.min()).item()
                g_std = cand_gap.std().item()

                if g_range > 0.001 and g_std > 0.0001:
                    cand_gap_norm = (cand_gap - cand_gap.min()) / (g_range + 1e-8)
                    cand_scores = I_final[candidate_indices].clone()
                    cand_I_std = max(cand_scores.std().item(), 1e-8)

                    b_warmup = min(1.0, float(current_rd) / 3.0)
                    raw_lambda = float(self.lambda_bias) * b_warmup
                    gn_std = cand_gap_norm.std().item()

                    if gn_std > 1e-6:
                        effect_ratio = raw_lambda * gn_std / cand_I_std
                        if effect_ratio > CAP_RATIO:
                            effective_lambda_bias = CAP_RATIO * cand_I_std / gn_std
                        else:
                            effective_lambda_bias = raw_lambda
                    else:
                        effective_lambda_bias = raw_lambda

                    cand_scores = cand_scores + effective_lambda_bias * cand_gap_norm
                    sel_in_cand = torch.topk(cand_scores, k=min(n_select, n_candidate), largest=True).indices
                    sel = candidate_indices[sel_in_cand]
                    bias_active = True
                    n_bias_reranked = n_candidate

                    actual_effect = effective_lambda_bias * gn_std / cand_I_std
                    print(f"  [B-dir] RERANK(raw_clip) active: λ={effective_lambda_bias:.4f} "
                          f"candidates={n_candidate} "
                          f"gap: mean={cand_gap.mean():.4f} std={g_std:.4f} "
                          f"eff/cand_std={actual_effect:.3f}")
                else:
                    print(f"  [B-dir] GATED OFF raw (range={g_range:.6f} std={g_std:.6f})")
                    sel = candidate_indices[:n_select]
        else:
            if enable_bias_calibration:
                print(f"  [B-dir] WAITING (round={current_rd} < 1)")
            # B不启用时，直接 topk
            sel = torch.topk(I_final, k=min(n_select, I_final.numel()), largest=True).indices

        sel_np = sel.detach().cpu().numpy()

        if not return_metrics:
            return sel_np

        metrics = {
            "U_intra": float(U_intra.mean().item()),
            "U_inter": float(U_inter.mean().item()),
            "I_final": float(I_final.mean().item()),
            "bias": float(bias_raw.mean().item()),
            "density": float(density_raw.mean().item()),
            "density_log": float(density_norm.mean().item()),
            "score": float(I_final.mean().item()),
            "global_score": float(I_final.mean().item()),
            "effective_lambda_density": float(effective_lambda_density),
            "effective_lambda_bias": float(effective_lambda_bias),
            "density_ramp": float(warmup_ratio) if density_active else 0.0,
            "bias_active": bias_active,
            "density_active": density_active,
            "n_bias_reranked": n_bias_reranked,
        }
        return sel_np, metrics
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
    #         labeled_features: Optional[torch.Tensor] = None,
    #         labeled_labels: Optional[torch.Tensor] = None,
    #         projected_img: Optional[torch.Tensor] = None,
    #         projected_txt: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     v20 采样函数
    #     U = 超球面vMF不确定性（核心）
    #     B = 跨模态方向差异（辅助，超球面测地距离）
    #     D = 类别覆盖多样性（辅助，基于超球面类别锚点）
    #
    #     新增参数:
    #         labeled_labels: [M, C] 已标注样本的多标签矩阵，用于D模块的类别频率统计
    #     """
    #     img_features = F.normalize(img_features, dim=1)
    #     txt_features = F.normalize(txt_features, dim=1)
    #     fused = F.normalize(0.5 * (img_features + txt_features), dim=1)
    #     N = img_features.shape[0]
    #
    #     if label_emb is None:
    #         label_emb = self.label_emb
    #
    #     # ===== U 模块（核心，完全不变）=====
    #     mu_i, k_i = self._vmf_params(img_features, label_emb)
    #     mu_t, k_t = self._vmf_params(txt_features, label_emb)
    #
    #     U_intra = 0.5 * (self._intra_uncertainty(k_i) + self._intra_uncertainty(k_t))
    #     U_inter = self._inter_uncertainty(mu_i, k_i, mu_t, k_t)
    #     I_sample = self.alpha_sample * U_intra + (1.0 - self.alpha_sample) * U_inter
    #
    #     # 冷启动 warmup
    #     if round_id is not None and round_id <= 2:
    #         warmup_blend = float(round_id) / 3.0
    #         random_scores = torch.rand_like(I_sample)
    #         I_sample = warmup_blend * I_sample + (1.0 - warmup_blend) * random_scores
    #
    #     current_rd = round_id if round_id is not None else 0
    #     I_std = max(I_sample.std().item(), 1e-8)
    #
    #     I_final = I_sample.clone()
    #     CAP_RATIO = 0.12
    #
    #     # =============================================================
    #     # B 模块：跨模态方向差异（Bi-modal Direction Gap）
    #     #
    #     # 超球面解释：
    #     #   img_features 和 txt_features 都在单位超球面上（已归一化）。
    #     #   1 - cos(img, txt) = 1 - <μ_img, μ_txt>
    #     #   这等价于超球面上两个方向的 Versine 距离：
    #     #     versine(θ) = 1 - cos(θ)
    #     #   其中 θ 是 μ_img 和 μ_txt 之间的测地距离（弧长 = arccos）。
    #     #
    #     # 与 U_inter 的区别和互补：
    #     #   U_inter 衡量的是两个 vMF 分布的 overlap，
    #     #   同时依赖方向差异 AND κ 大小。
    #     #   B 模块只看方向差异，不依赖 κ 估计。
    #     #   当 label_emb 不成熟导致 κ 不准时，B 提供稳定的补充信号。
    #     #
    #     # 消融保证：
    #     #   - 信号范围 [0, ~2]，方差充足（比旧版 |u_img-u_txt| 强 ~50 倍）
    #     #   - 不依赖 label_emb → 第1轮起即可启用
    #     #   - 方向确定：gap大 → 标注价值高（三个数据集均成立）
    #     #   - 后期：大部分样本 gap 小，少数难样本 gap 大 → 精准聚焦
    #     # =============================================================
    #     effective_lambda_bias = 0.0
    #     bias_raw = torch.zeros(N, device=fused.device)
    #     bias_norm = torch.zeros(N, device=fused.device)
    #     bias_active = False
    #     n_bias_filtered = 0
    #
    #     if enable_bias_calibration and current_rd >= 1:
    #         # 超球面 versine 距离
    #         direction_gap = 1.0 - (img_features * txt_features).sum(dim=1)  # [N]
    #         bias_raw = direction_gap
    #
    #         b_range = (bias_raw.max() - bias_raw.min()).item()
    #         b_std = bias_raw.std().item()
    #
    #         if b_range > 0.001 and b_std > 0.0001:
    #             bias_norm = (bias_raw - bias_raw.min()) / (b_range + 1e-8)
    #             bn_std = bias_norm.std().item()
    #
    #             # round 1 用 0.5 力度, round 2 用 0.75, round 3+ 满力度
    #             b_warmup = min(1.0, float(current_rd) / 3.0)
    #
    #             raw_lambda = float(self.lambda_bias) * b_warmup
    #             if bn_std > 1e-6:
    #                 effect_ratio = raw_lambda * bn_std / I_std
    #                 if effect_ratio > CAP_RATIO:
    #                     effective_lambda_bias = CAP_RATIO * I_std / bn_std
    #                 else:
    #                     effective_lambda_bias = raw_lambda
    #             else:
    #                 effective_lambda_bias = raw_lambda
    #
    #             I_final = I_final + effective_lambda_bias * bias_norm
    #             bias_active = True
    #
    #             actual_effect = effective_lambda_bias * bn_std / I_std if bn_std > 1e-6 else 0.0
    #             print(f"  [B-dir] ACTIVE λ={effective_lambda_bias:.4f} "
    #                   f"(cfg={self.lambda_bias} warmup={b_warmup:.2f}) "
    #                   f"gap: mean={direction_gap.mean():.4f} std={b_std:.4f} "
    #                   f"range=[{direction_gap.min():.4f},{direction_gap.max():.4f}] "
    #                   f"eff/I_std={actual_effect:.3f}")
    #         else:
    #             print(f"  [B-dir] GATED OFF (range={b_range:.6f} std={b_std:.6f})")
    #     elif enable_bias_calibration:
    #         print(f"  [B-dir] WAITING (round={current_rd} < 1)")
    #
    #     # =============================================================
    #     # D 模块：类别覆盖多样性（Class Coverage Diversity）
    #     #
    #     # 超球面解释：
    #     #   label_emb[c] 是类别 c 在超球面上的方向锚点。
    #     #   样本对类别 c 的亲和度 = softmax(<f, label_emb[c]> / τ)
    #     #   这衡量了样本在超球面上与各类别方向的接近程度。
    #     #   给"预测为稀缺类"的样本加分 → 平衡超球面上的类别覆盖。
    #     #
    #     # 长尾保护（解决你的疑问）：
    #     #   1. 用 1/sqrt(freq) 而非 1/freq，压制极端权重
    #     #   2. 样本数 < MIN_SAMPLES_FOR_WEIGHT 的类不给额外权重
    #     #   3. class_weight 做 clamp，防止任何单类权重超过均值的 5 倍
    #     #
    #     # 消融保证：
    #     #   - 不会退化：标注池越大 → 频率估计越准 → 信号越好
    #     #   - mAP 是类别平均 → 稀缺类提升对 mAP 贡献大
    #     #   - 三个数据集都有类别不平衡
    #     # =============================================================
    #     effective_lambda_density = 0.0
    #     density_raw = torch.zeros(N, device=fused.device)
    #     density_norm = torch.zeros(N, device=fused.device)
    #     density_active = False
    #     warmup_ratio = 0.0
    #
    #     if enable_diversity_selection and current_rd >= 1:
    #         has_labeled = (labeled_labels is not None and labeled_labels.shape[0] >= 10)
    #         has_label_emb = (label_emb is not None and self.label_emb_initialized is not None
    #                          and self.label_emb_initialized.sum().item() >= max(2, self.num_classes // 2))
    #
    #         if has_labeled and has_label_emb:
    #             y_lb = labeled_labels.float()
    #             M = y_lb.shape[0]
    #             C = y_lb.shape[1]
    #
    #             # 1. 类别频率
    #             class_count = y_lb.sum(dim=0)  # [C]
    #             class_freq = class_count / max(float(M), 1.0)  # [C]
    #
    #             # 2. 稀缺类权重（sqrt 倒数 + 长尾保护）
    #             MIN_SAMPLES_FOR_WEIGHT = 3  # 少于3个样本的类不给额外权重
    #             eps_freq = 1e-4
    #             class_weight = 1.0 / torch.sqrt(class_freq + eps_freq)  # [C]
    #
    #             # 长尾保护1：样本数太少的类，权重设为1（不给额外激励）
    #             sparse_mask = class_count < MIN_SAMPLES_FOR_WEIGHT
    #             if sparse_mask.any():
    #                 n_sparse = int(sparse_mask.sum().item())
    #                 print(f"  [D-cls] 长尾保护: {n_sparse} classes with <{MIN_SAMPLES_FOR_WEIGHT} samples → weight=1")
    #
    #             # 未初始化的类也不给权重
    #             if self.label_emb_initialized is not None:
    #                 uninit_mask = ~self.label_emb_initialized
    #                 sparse_mask = sparse_mask | uninit_mask
    #
    #             # 对有效类做归一化（均值=1），稀疏/未初始化类设为1
    #             valid_mask = ~sparse_mask
    #             if valid_mask.sum() > 0:
    #                 valid_weights = class_weight[valid_mask]
    #                 valid_mean = valid_weights.mean().clamp_min(1e-8)
    #                 class_weight[valid_mask] = valid_weights / valid_mean
    #             class_weight[sparse_mask] = 1.0
    #
    #             # 长尾保护2：clamp 防止单类权重超过 5 倍
    #             class_weight = class_weight.clamp(0.2, 5.0)
    #
    #             # 3. 未标注样本对各类的亲和度（超球面上的方向接近度）
    #             le_norm = F.normalize(label_emb, dim=1)
    #             sim_to_class = fused @ le_norm.t()  # [N, C]
    #
    #             tau_d = max(float(self.kappa_soft_tau), 0.05)
    #             affinity = torch.softmax(sim_to_class / tau_d, dim=1)  # [N, C]
    #
    #             # 4. 加权得分
    #             diversity_score = (affinity * class_weight.unsqueeze(0)).sum(dim=1)  # [N]
    #             density_raw = diversity_score
    #
    #             d_range = (density_raw.max() - density_raw.min()).item()
    #             d_std = density_raw.std().item()
    #
    #             if d_range > 0.001 and d_std > 0.0001:
    #                 density_norm = (density_raw - density_raw.min()) / (d_range + 1e-8)
    #                 dn_std = density_norm.std().item()
    #
    #                 _warmup = float(self.density_warmup_rounds) if self.density_warmup_rounds > 0 else 3.0
    #                 warmup_ratio = min(1.0, float(current_rd) / _warmup)
    #
    #                 raw_lambda = float(self.lambda_density) * warmup_ratio
    #                 if dn_std > 1e-6:
    #                     effect_ratio = raw_lambda * dn_std / I_std
    #                     if effect_ratio > CAP_RATIO:
    #                         effective_lambda_density = CAP_RATIO * I_std / dn_std
    #                     else:
    #                         effective_lambda_density = raw_lambda
    #                 else:
    #                     effective_lambda_density = raw_lambda
    #
    #                 I_final = I_final + effective_lambda_density * density_norm
    #                 density_active = True
    #
    #                 freq_valid = class_freq[class_freq > eps_freq]
    #                 imbalance_ratio = (freq_valid.max() / freq_valid.min()).item() if freq_valid.numel() > 1 else 1.0
    #
    #                 actual_effect = effective_lambda_density * dn_std / I_std if dn_std > 1e-6 else 0.0
    #                 print(f"  [D-cls] ACTIVE λ={effective_lambda_density:.4f} "
    #                       f"(cfg={self.lambda_density} warm={self.density_warmup_rounds}) "
    #                       f"class_imbalance={imbalance_ratio:.1f}x "
    #                       f"M={M} C_init={int(self.label_emb_initialized.sum().item())}/{C} "
    #                       f"sparse={int(sparse_mask.sum().item())} "
    #                       f"warmup={warmup_ratio:.2f} "
    #                       f"eff/I_std={actual_effect:.3f}")
    #             else:
    #                 print(f"  [D-cls] GATED OFF (range={d_range:.6f} std={d_std:.6f})")
    #         else:
    #             n_lb = 0 if labeled_labels is None else labeled_labels.shape[0]
    #             n_init = 0 if self.label_emb_initialized is None else int(self.label_emb_initialized.sum().item())
    #             print(f"  [D-cls] SKIP: labeled={n_lb}, label_emb_init={n_init}/{self.num_classes}")
    #
    #     # ===== 统一 topk =====
    #     sel = torch.topk(I_final, k=min(n_select, I_final.numel()), largest=True).indices
    #     sel_np = sel.detach().cpu().numpy()
    #
    #     score = I_final.clone()
    #     score[sel] += 1.0
    #
    #     if not return_metrics:
    #         return sel_np
    #
    #     metrics = {
    #         "U_intra": float(U_intra.mean().item()),      # 标量
    #         "U_inter": float(U_inter.mean().item()),      # 标量
    #         "I_final": float(I_final.mean().item()),      # 标量
    #         "bias": float(bias_raw.mean().item()),         # 标量
    #         "density": float(density_raw.mean().item()),   # 标量
    #         "density_log": float(density_norm.mean().item()),
    #         "score": float(I_final.mean().item()),
    #         "global_score": float(I_final.mean().item()),
    #         "effective_lambda_density": float(effective_lambda_density),
    #         "effective_lambda_bias": float(effective_lambda_bias),
    #         "density_ramp": float(warmup_ratio) if density_active else 0.0,
    #         "bias_active": bias_active,
    #         "density_active": density_active,
    #         "n_bias_filtered": n_bias_filtered,
    #     }
    #     return sel_np, metrics

        # if not return_metrics:
        #     return sel_np
        #
        # metrics = {
        #     "U_intra": U_intra.detach().cpu().numpy(),
        #     "U_inter": U_inter.detach().cpu().numpy(),
        #     "I_final": I_final.detach().cpu().numpy(),
        #     "bias": bias_raw.detach().cpu().numpy(),
        #     "density": density_raw.detach().cpu().numpy(),
        #     "density_log": density_norm.detach().cpu().numpy(),
        #     "score": score.detach().cpu().numpy(),
        #     "global_score": I_final.detach().cpu().numpy(),
        #     "effective_lambda_density": float(effective_lambda_density),
        #     "effective_lambda_bias": float(effective_lambda_bias),
        #     "density_ramp": float(warmup_ratio) if density_active else 0.0,
        #     "bias_active": bias_active,
        #     "density_active": density_active,
        #     "n_bias_filtered": n_bias_filtered,
        # }
        # return sel_np, metrics

    # ================================================================
    # 核心方法2：compute_training_weights（v20 重写）
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
    ) -> torch.Tensor:
        """
        v21: 难度自适应权重

        与 v20 的区别：pan_losses 现在是当前轮训练预热阶段收集的，
        而不是上一轮的历史数据。信号更准确。
        """
        img_features = F.normalize(img_features, dim=1)
        txt_features = F.normalize(txt_features, dim=1)
        N = img_features.shape[0]

        if N < 2:
            return torch.ones(N, device=img_features.device)

        SCALE = min(max(float(self.train_weight_scale), 0.05), 0.50)

        use_pan = (pan_losses is not None and pan_losses.numel() == N)

        if use_pan:
            raw = pan_losses.float().to(img_features.device)

            raw_mean = raw.mean()
            raw_std = raw.std().clamp_min(1e-6)
            raw = raw.clamp(raw_mean - 3.0 * raw_std, raw_mean + 3.0 * raw_std)

            r_min = raw.min()
            r_max = raw.max()
            r_range = (r_max - r_min).item()

            if r_range > 1e-6:
                normalized = (raw - r_min) / (r_range + 1e-8)
            else:
                print(f"  [W-pan] PAN loss range too small ({r_range:.6f}) → 等权")
                return torch.ones(N, device=img_features.device)

            k_sigmoid = 4.0
            weight_raw = torch.sigmoid(k_sigmoid * (normalized - 0.5))

            w = 1.0 + SCALE * (2.0 * weight_raw - 1.0)

            signal_name = "PAN-loss"
            signal_mean = raw_mean.item()
            signal_std = raw_std.item()

        else:
            alignment = (img_features * txt_features).sum(dim=1)
            align_mean = alignment.mean()
            align_std = alignment.std().clamp_min(1e-6)

            difficulty = -alignment

            d_min = difficulty.min()
            d_max = difficulty.max()
            d_range = (d_max - d_min).item()

            if d_range > 1e-6:
                normalized = (difficulty - d_min) / (d_range + 1e-8)
            else:
                print(f"  [W-align] alignment range too small ({d_range:.6f}) → 等权")
                return torch.ones(N, device=img_features.device)

            k_sigmoid = 4.0
            weight_raw = torch.sigmoid(k_sigmoid * (normalized - 0.5))
            w = 1.0 + SCALE * (2.0 * weight_raw - 1.0)

            signal_name = "alignment(fallback)"
            signal_mean = align_mean.item()
            signal_std = align_std.item()

        w_min = max(0.5, 1.0 - SCALE)
        w_max = min(2.0, 1.0 + SCALE)
        w = w.clamp(w_min, w_max)

        w = w / (w.mean() + 1e-8)

        print(f"  [W-diff] signal={signal_name} "
              f"mean={signal_mean:.4f} std={signal_std:.4f} | "
              f"SCALE={SCALE:.3f}(cfg={self.train_weight_scale}) | "
              f"w: {w.mean():.4f}±{w.std():.4f} "
              f"[{w.min():.3f},{w.max():.3f}]")

        return w