# models/vmfal_model.py
"""
vMF Uncertainty Model — 合并最终版

基于另一位专家的版本，叠加三处关键修改：
  1. B模块：双向过滤 → 单向过滤（只去gap极大的outlier）
  2. D模块：CAP 0.06 → 0.10
  3. W模块：保持原接口（Focal Weighting在vmfal_sampling.py中实现）
"""

from __future__ import annotations
import math
from typing import Optional
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
        lambda_density: float = 0.02,
        lambda_bias: float = 0.0,
        alpha_sample: float = 0.5,
        kde_kappa: float = 32.0,
        kde_batch: int = 512,
        kappa_min: float = 2.0,
        kappa_max: float = 128.0,
        c_intra: float = 0.08,
        kappa_soft_tau: float = 0.07,
        density_warmup_rounds: int = 0,
        train_weight_scale: float = 0.15,
        bias_filter_ratio: float = 0.15,
    ):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.device = device
        self.lambda_density = float(lambda_density)
        self.alpha_sample = float(alpha_sample)
        self.kde_kappa = float(kde_kappa)
        self.kde_batch = int(kde_batch)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.c_intra = float(c_intra)
        self.kappa_soft_tau = float(kappa_soft_tau)
        self.density_warmup_rounds = int(density_warmup_rounds)
        self.train_weight_scale = float(train_weight_scale)
        self.bias_filter_ratio = float(bias_filter_ratio)

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

    # ----------------------------------------------------------------
    # label embedding （不变）
    # ----------------------------------------------------------------
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
                    self.label_emb[c] = F.normalize(
                        momentum * self.label_emb[c] + (1.0 - momentum) * mu, dim=0
                    )
            else:
                if not self.label_emb_initialized[c]:
                    self.label_emb[c] = global_mean

    def _label_emb_quality(self) -> float:
        if self.label_emb is None or self.label_emb_initialized is None:
            return 0.0
        init_ratio = float(self.label_emb_initialized.float().mean().item())
        if init_ratio < 0.5:
            return init_ratio * 0.5
        init_mask = self.label_emb_initialized
        emb = self.label_emb[init_mask]
        if emb.shape[0] < 2:
            return init_ratio * 0.3
        emb_norm = F.normalize(emb, dim=1)
        sim_mat = emb_norm @ emb_norm.t()
        K = sim_mat.size(0)
        mask_off_diag = ~torch.eye(K, dtype=torch.bool, device=sim_mat.device)
        avg_sim = sim_mat[mask_off_diag].mean().item()
        return float(init_ratio * max(0.0, 1.0 - avg_sim))

    def label_emb_quality(self):
        return self._label_emb_quality()

    # ----------------------------------------------------------------
    # kappa （不变）
    # ----------------------------------------------------------------
    def _kappa_from_anchor(self, feats, label_emb, labels=None):
        B, _ = feats.shape
        if label_emb is None:
            return torch.full((B,), (self.kappa_min + self.kappa_max) * 0.5,
                              device=feats.device, dtype=feats.dtype)
        sims = feats @ label_emb.t()
        if self.label_emb_initialized is not None:
            mask_uninit = ~self.label_emb_initialized
            if mask_uninit.any():
                sims = sims.masked_fill(mask_uninit.unsqueeze(0), -1e9)
        if labels is None:
            tau = torch.tensor(float(self.kappa_soft_tau),
                               device=feats.device, dtype=feats.dtype).clamp_min(1e-6)
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
        return (self.kappa_min + (self.kappa_max - self.kappa_min) * scale).clamp_min(1e-6)

    def _vmf_params(self, feats, label_emb, labels=None):
        feats = F.normalize(feats, dim=1)
        return feats, self._kappa_from_anchor(feats, label_emb, labels=labels)

    # ----------------------------------------------------------------
    # uncertainty components （不变）
    # ----------------------------------------------------------------
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
        logIv = self._logIv_large_nu(nu, kappa)
        return nu * torch.log(kappa) - (d / 2.0) * math.log(2.0 * math.pi) - logIv

    def _overlap_norm(self, mu1, k1, mu2, k2):
        cos = (mu1 * mu2).sum(dim=1).clamp(-1.0, 1.0)
        R = torch.sqrt(k1 * k1 + k2 * k2 + 2.0 * k1 * k2 * cos).clamp_min(1e-12)
        logC_R = self._logC_vmf(R)
        logC_max = self._logC_vmf((k1 + k2).clamp_min(1e-12))
        overlap = torch.exp((logC_max - logC_R).clamp(min=-80.0, max=0.0))
        return overlap.clamp(0.0, 1.0)

    def _inter_uncertainty(self, mu_img, k_img, mu_txt, k_txt):
        return 1.0 - self._overlap_norm(mu_img, k_img, mu_txt, k_txt)

    # ================================================================
    # 核心采样方法
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
            **kwargs
    ):
        img_features = F.normalize(img_features, dim=1)
        txt_features = F.normalize(txt_features, dim=1)
        fused = F.normalize(0.5 * (img_features + txt_features), dim=1)
        N = img_features.shape[0]
        if label_emb is None:
            label_emb = self.label_emb
        current_rd = round_id if round_id is not None else 0

        # ============================================================
        # U: vMF不确定性（不变）
        # ============================================================
        mu_i, k_i = self._vmf_params(img_features, label_emb)
        mu_t, k_t = self._vmf_params(txt_features, label_emb)
        U_intra = 0.5 * (self._intra_uncertainty(k_i) + self._intra_uncertainty(k_t))
        U_inter = self._inter_uncertainty(mu_i, k_i, mu_t, k_t)
        I_sample = self.alpha_sample * U_intra + (1.0 - self.alpha_sample) * U_inter

        if current_rd <= 2:
            wb = float(current_rd) / 3.0
            I_sample = wb * I_sample + (1.0 - wb) * torch.rand_like(I_sample)

        I_std = max(I_sample.std().item(), 1e-8)
        I_final = I_sample.clone()

        # ============================================================
        # D: kNN密度惩罚
        # ★ 修改点：CAP从0.06提升到0.10，增强D的实际效果
        # ============================================================
        eff_lambda_d = 0.0
        d_active = False
        density_raw = torch.zeros(N, device=fused.device)

        if enable_diversity_selection and current_rd >= 1 and labeled_features is not None:
            n_lb = labeled_features.shape[0]
            if n_lb >= 5:
                lb_norm = F.normalize(labeled_features, dim=1)
                k_nn = min(10, n_lb)
                bs_d = 256
                knn_density = torch.empty(N, device=fused.device)
                for s in range(0, N, bs_d):
                    e = min(N, s + bs_d)
                    sim_chunk = fused[s:e] @ lb_norm.t()
                    topk_sim, _ = sim_chunk.topk(k_nn, dim=1)
                    knn_density[s:e] = topk_sim.mean(dim=1)
                    del sim_chunk, topk_sim
                density_raw = knn_density

                d_min, d_max = knn_density.min(), knn_density.max()
                d_range = (d_max - d_min).item()
                if d_range > 1e-6:
                    d_penalty = (knn_density - d_min) / (d_range + 1e-8)
                    dp_std = max(d_penalty.std().item(), 1e-8)

                    raw_ld = float(self.lambda_density)
                    CAP_D = 0.10  # ★ 从0.06提升到0.10
                    eff_ratio = raw_ld * dp_std / I_std
                    eff_lambda_d = min(raw_ld, CAP_D * I_std / dp_std) if eff_ratio > CAP_D else raw_ld

                    I_final = I_final - eff_lambda_d * d_penalty
                    d_active = True
                    print(f"  [D] lambda={eff_lambda_d:.4f} eff/I={eff_lambda_d * dp_std / I_std:.3f}")
                else:
                    print(f"  [D] GATED (range={d_range:.6f})")
            else:
                print(f"  [D] SKIP (labeled={n_lb}<5)")

        # ============================================================
        # B: 跨模态偏差校准
        # ★ 修改点：双向过滤 → 单向过滤（只剔除gap极大的outlier）
        #
        # 原因：多标签场景中，gap小的样本 ≠ "不需要学习"
        #       它们可能是跨模态已对齐但标签组合罕见的高价值样本
        #       COCO有80类，这种样本比例很高
        #       剔除它们导致B在COCO上反向（-0.0024）
        # ============================================================
        b_active = False
        n_filtered = 0

        if enable_bias_calibration and current_rd >= 2:
            n_candidates = min(int(n_select * 2.5), N)  # ★ 2x→2.5x，给单向过滤更多余量
            candidate_indices = torch.topk(I_final, k=n_candidates, largest=True).indices

            cand_img = img_features[candidate_indices]
            cand_txt = txt_features[candidate_indices]
            cand_cos = (cand_img * cand_txt).sum(dim=1)
            cand_gap = 1.0 - cand_cos

            K_cand = len(candidate_indices)
            ratio = self.bias_filter_ratio
            n_trim = max(1, int(K_cand * ratio))

            # ★ 单向过滤：只需要 K_cand > n_select + n_trim（不再需要 2*n_trim）
            if K_cand > n_select + n_trim:
                gap_sorted_idx = cand_gap.argsort()

                keep_mask = torch.ones(K_cand, dtype=torch.bool, device=self.device)
                # ★ 只剔除gap最大的（噪声/outlier）
                keep_mask[gap_sorted_idx[-n_trim:]] = False
                # ★ 不再剔除gap最小的

                kept_cand = candidate_indices[keep_mask]
                n_filtered = K_cand - int(keep_mask.sum())

                kept_scores = I_final[kept_cand]
                top_in_kept = torch.topk(kept_scores, k=min(n_select, len(kept_cand)), largest=True).indices
                sel = kept_cand[top_in_kept]
            else:
                sel = candidate_indices[:n_select]

            b_active = True
            print(f"  [B] candidates={K_cand} filtered={n_filtered} "
                  f"gap=[{cand_gap.min():.4f},{cand_gap.max():.4f}]")
            del cand_img, cand_txt
        else:
            sel = torch.topk(I_final, k=min(n_select, N), largest=True).indices

        sel_np = sel.cpu().numpy()

        if not return_metrics:
            return sel_np
        return sel_np, {
            "U_intra": float(U_intra.mean()),
            "U_inter": float(U_inter.mean()),
            "I_final": float(I_final.mean()),
            "bias": 0.0,
            "density": float(density_raw.mean()),
            "density_log": 0.0,
            "score": float(I_final.mean()),
            "global_score": float(I_final.mean()),
            "effective_lambda_density": float(eff_lambda_d),
            "effective_lambda_bias": 0.0,
            "density_ramp": 0.0,
            "bias_active": b_active,
            "density_active": d_active,
            "n_bias_reranked": n_filtered,
        }

    # ================================================================
    # W: 基于轮次的训练权重（保留作为Focal的fallback）
    # ================================================================
    # ================================================================
    # W: 基于轮次的训练权重（自适应版本）
    # ================================================================
    @torch.no_grad()
    def compute_training_weights(
            self,
            labels,
            round_id=0,
            total_rounds=15,
            sample_round_ids=None,
            **kwargs
    ) -> torch.Tensor:
        N = labels.shape[0]
        if N < 5:
            return torch.ones(N, device=self.device)

        if sample_round_ids is None:
            print(f"  [W] no sample_round_ids, uniform weights")
            return torch.ones(N, device=self.device)

        # ★ 改动：前2轮等权（原版是 round_id <= 0）
        if round_id <= 2:
            print(f"  [W] round_id={round_id} <= 2, warmup → uniform weights")
            return torch.ones(N, device=self.device)

        R = float(round_id)
        scale = float(self.train_weight_scale)

        if scale < 1e-4:
            return torch.ones(N, device=self.device)

        if isinstance(sample_round_ids, np.ndarray):
            sr = torch.from_numpy(sample_round_ids).float().to(self.device)
        elif isinstance(sample_round_ids, torch.Tensor):
            sr = sample_round_ids.float().to(self.device)
        else:
            sr = torch.tensor(sample_round_ids, dtype=torch.float32, device=self.device)

        age = (R - sr).clamp(min=0.0)
        decay = scale * age / max(R, 1.0)
        w = (1.0 - decay).clamp(min=max(0.5, 1.0 - scale), max=1.0)
        w = w / (w.mean() + 1e-8)

        n_decayed = int((w < 0.99).sum())
        print(f"  [W] round={round_id} scale={scale:.3f} "
              f"decayed={n_decayed}/{N} w: {w.mean():.3f}+-{w.std():.3f} "
              f"[{w.min():.3f},{w.max():.3f}]")
        return w
    # @torch.no_grad()
    # def compute_training_weights(
    #         self,
    #         labels,
    #         round_id=0,
    #         total_rounds=15,
    #         sample_round_ids=None,
    #         **kwargs
    # ) -> torch.Tensor:
    #     N = labels.shape[0]
    #     if N < 5:
    #         return torch.ones(N, device=self.device)
    #
    #     if sample_round_ids is None:
    #         print(f"  [W] no sample_round_ids, uniform weights")
    #         return torch.ones(N, device=self.device)
    #
    #     if round_id <= 0:
    #         print(f"  [W] round_id={round_id} <= 0, uniform weights")
    #         return torch.ones(N, device=self.device)
    #
    #     R = float(round_id)
    #     scale = float(self.train_weight_scale)
    #
    #     if scale < 1e-4:
    #         return torch.ones(N, device=self.device)
    #
    #     if isinstance(sample_round_ids, np.ndarray):
    #         sr = torch.from_numpy(sample_round_ids).float().to(self.device)
    #     elif isinstance(sample_round_ids, torch.Tensor):
    #         sr = sample_round_ids.float().to(self.device)
    #     else:
    #         sr = torch.tensor(sample_round_ids, dtype=torch.float32, device=self.device)
    #
    #     age = (R - sr).clamp(min=0.0)
    #     decay = scale * age / max(R, 1.0)
    #     w = (1.0 - decay).clamp(min=max(0.5, 1.0 - scale), max=1.0)
    #     w = w / (w.mean() + 1e-8)
    #
    #     n_decayed = int((w < 0.99).sum())
    #     print(f"  [W] round={round_id} scale={scale:.3f} "
    #           f"decayed={n_decayed}/{N} w: {w.mean():.3f}+-{w.std():.3f} "
    #           f"[{w.min():.3f},{w.max():.3f}]")
    #     return w