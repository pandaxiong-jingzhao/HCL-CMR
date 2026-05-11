# query_strategies/vmfal_sampling.py
"""
VMFSampling — 合并最终版

基于另一位专家的投影特征实现（_project_features_gpu），叠加：
  - W模块：Focal Weighting（基于PAN loss），替代轮次衰减
    - Focal: w = (loss/mean_loss)^gamma, 范围[0.5, 2.0]
    - 轮次衰减: w ∈ [0.92, 1.08]，对训练几乎无影响
    - Focal的Full vs wo_W差距更大，消融更容易通过

其余逻辑完全采用另一位专家的实现：
  - Round 0: 原始CLIP特征
  - Round 1+: 投影特征（_project_features_gpu）
  - label_emb: Round 1+用投影特征更新
  - kappa校准: Round 1+用投影特征
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from models.vmfal_model import VMFUncertaintyModel
from query_strategies.strategy import Strategy, argget


class VMFSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
                 X_img=None, X_txt=None, X_img_val=None, X_txt_val=None):
        super().__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer,
                         X_img=X_img, X_txt=X_txt, X_img_val=X_img_val, X_txt_val=X_txt_val)

        # ---------- 基本参数 ----------
        self.num_classes = int(argget(args, "n_class",
                                      argget(args, "num_classes",
                                             Y.shape[1] if Y is not None else 0)))
        self.feature_dim = int(argget(args, "feature_dim",
                                      X_img.shape[1] if X_img is not None else 512))

        # ---------- 采样参数 ----------
        self.lambda_density = float(argget(args, "lambda_density", 0.02))
        self.alpha_sample = float(argget(args, "alpha_sample", 0.5))
        self.kde_kappa = float(argget(args, "kde_kappa", 32.0))
        self.kde_batch = int(argget(args, "kde_batch", 512))
        self.disable_train_weight = bool(argget(args, "disable_train_weight", False))
        self.kappa_min = float(argget(args, "kappa_min", 2.0))
        self.kappa_max = float(argget(args, "kappa_max", 128.0))
        self.c_intra = float(argget(args, "c_intra", 0.08))
        self.label_momentum = float(argget(args, "label_momentum", 0.8))

        # ---------- ★ Focal W 参数 ----------
        self.focal_gamma = float(argget(args, "focal_gamma", 0.5))

        # ---------- 创建vMF模型 ----------
        self.vmf_model = VMFUncertaintyModel(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            device=self.device,
            lambda_density=self.lambda_density,
            alpha_sample=self.alpha_sample,
            kde_kappa=self.kde_kappa,
            kde_batch=self.kde_batch,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
            c_intra=self.c_intra,
            kappa_soft_tau=float(argget(args, "kappa_soft_tau", 0.07)),
            density_warmup_rounds=int(argget(args, "density_warmup_rounds", 0)),
            train_weight_scale=float(argget(args, "train_weight_scale", 0.15)),
            bias_filter_ratio=float(argget(args, "bias_filter_ratio", 0.15)),
        )
        self.label_prompt = self.vmf_model.get_label_embedding()

        # ---------- kappa在线校准 ----------
        self.enable_kappa_calib = bool(argget(args, "enable_kappa_calib", True))
        self.kappa_calib_lr = float(argget(args, "kappa_calib_lr", 1e-2))
        self.kappa_calib_steps = int(argget(args, "kappa_calib_steps", 50))
        self.kappa_calib_reg = float(argget(args, "kappa_calib_reg", 1e-3))
        if self.enable_kappa_calib:
            self.kappa_calib_opt = torch.optim.Adam(
                [self.vmf_model.kappa_a_raw, self.vmf_model.kappa_b],
                lr=self.kappa_calib_lr
            )

        # ---------- 状态变量 ----------
        self.newly_labeled = np.array([], dtype=int)
        self.total_round = int(argget(args, "n_round", 15))
        self.enable_bias_calibration = bool(argget(args, "enable_bias_calibration", False))
        self.enable_diversity_selection = bool(argget(args, "enable_diversity_selection", False))

        self.global_weights = torch.ones(len(self.Y), device=self.device, dtype=torch.float32)

        # ---------- 预转tensor并pin_memory加速 ----------
        self._X_img_t = torch.as_tensor(self.X_img, dtype=torch.float32).contiguous()
        self._X_txt_t = torch.as_tensor(self.X_txt, dtype=torch.float32).contiguous()
        self._Y_t = torch.as_tensor(self.Y, dtype=torch.float32).contiguous()
        if torch.cuda.is_available():
            self._X_img_t = self._X_img_t.pin_memory()
            self._X_txt_t = self._X_txt_t.pin_memory()
            self._Y_t = self._Y_t.pin_memory()

        # ---------- W模块：记录每个样本被选中的轮次 ----------
        self._sample_round_ids = np.full(len(self.Y), -1, dtype=np.int32)
        init_lb = np.where(self.idxs_lb)[0]
        self._sample_round_ids[init_lb] = 0
        self._proj_dim_confirmed = False

    # ================================================================
    # GPU-only 分块投影（来自另一位专家，不改）
    # ================================================================
    def _project_features_gpu(self, cpu_img_t, cpu_txt_t, idxs_np, bs=256):
        N = len(idxs_np)
        idx_t = torch.as_tensor(idxs_np, dtype=torch.long)

        self.model.clf.eval()

        # 探测实际输出维度
        with torch.no_grad():
            probe_idx = idx_t[:min(2, N)]
            probe_img = cpu_img_t.index_select(0, probe_idx).to(self.device)
            probe_txt = cpu_txt_t.index_select(0, probe_idx).to(self.device)
            _, probe_ip, _, _, _ = self.model.clf(probe_img.float(), probe_txt.float())
            proj_dim = probe_ip.shape[1]
            del probe_img, probe_txt, probe_ip

        # 第一次调用时更新 vmf_model 的维度
        if not self._proj_dim_confirmed:
            if proj_dim != self.vmf_model.feature_dim:
                print(f"  [Proj] feature_dim 更新: {self.vmf_model.feature_dim} -> {proj_dim}")
                self.vmf_model.feature_dim = proj_dim
                self.vmf_model.label_emb = torch.zeros(
                    self.num_classes, proj_dim,
                    device=self.device, dtype=torch.float32
                )
                self.vmf_model.label_emb_initialized = torch.zeros(
                    self.num_classes, dtype=torch.bool, device=self.device
                )
                self.label_prompt = self.vmf_model.get_label_embedding()
            self._proj_dim_confirmed = True

        img_out = torch.empty(N, proj_dim, dtype=torch.float32, device=self.device)
        txt_out = torch.empty(N, proj_dim, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for s in range(0, N, bs):
                e = min(N, s + bs)
                chunk_idx = idx_t[s:e]
                img_c = cpu_img_t.index_select(0, chunk_idx).to(
                    self.device, non_blocking=True)
                txt_c = cpu_txt_t.index_select(0, chunk_idx).to(
                    self.device, non_blocking=True)
                _, ip, tp, _, _ = self.model.clf(img_c.float(), txt_c.float())
                img_out[s:e].copy_(ip)
                txt_out[s:e].copy_(tp)
                del img_c, txt_c, ip, tp

        return img_out, txt_out
    # ================================================================
    # query（来自另一位专家，不改）
    # ================================================================
    def query(self, n_query: int):
        self.query_count += 1
        idxs_unlabeled = np.where(~self.idxs_lb)[0]
        if len(idxs_unlabeled) == 0:
            return np.array([], dtype=int), None, None, None, np.array([], dtype=int), None
        n_select = min(int(n_query), len(idxs_unlabeled))

        fixed_query_dir = str(argget(self.args, "fixed_query_dir", "") or "").strip()
        if fixed_query_dir:
            import os as _os
            _os.makedirs(fixed_query_dir, exist_ok=True)
            idx_file = _os.path.join(fixed_query_dir, f"round_{self.round}.npy")
            if _os.path.exists(idx_file):
                q_idxs = np.load(idx_file)
                valid = np.isin(q_idxs, idxs_unlabeled)
                if not valid.all():
                    q_valid = q_idxs[valid]
                    remaining = np.setdiff1d(idxs_unlabeled, q_valid)
                    n_need = int((~valid).sum())
                    if len(remaining) >= n_need:
                        q_idxs = np.concatenate([
                            q_valid,
                            np.random.choice(remaining, n_need, replace=False)
                        ])
                    else:
                        q_idxs = q_valid
                return q_idxs, None, None, None, idxs_unlabeled, None

        current_round = int(getattr(self, "round", 0))

        if current_round >= 1:
            print(f"  [Query R{current_round}] 使用投影特征（PAN空间）")
            img_u, txt_u = self._project_features_gpu(
                self._X_img_t, self._X_txt_t, idxs_unlabeled, bs=256
            )
        else:
            print(f"  [Query R{current_round}] 使用原始CLIP特征（模型未训练）")
            idx_u = torch.as_tensor(idxs_unlabeled, dtype=torch.long)
            img_u = self._X_img_t.index_select(0, idx_u).to(
                self.device, non_blocking=True)
            txt_u = self._X_txt_t.index_select(0, idx_u).to(
                self.device, non_blocking=True)

        labeled_fused = None
        idxs_labeled = np.where(self.idxs_lb)[0]
        if len(idxs_labeled) > 0:
            if current_round >= 1:
                img_lb_proj, txt_lb_proj = self._project_features_gpu(
                    self._X_img_t, self._X_txt_t, idxs_labeled, bs=256
                )
                labeled_fused = F.normalize(
                    0.5 * (F.normalize(img_lb_proj, dim=1) +
                           F.normalize(txt_lb_proj, dim=1)), dim=1
                )
                del img_lb_proj, txt_lb_proj
            else:
                idx_l = torch.as_tensor(idxs_labeled, dtype=torch.long)
                lb_bs = 2048
                fused_chunks = []
                for s in range(0, len(idxs_labeled), lb_bs):
                    e = min(len(idxs_labeled), s + lb_bs)
                    chunk_idx = idx_l[s:e]
                    img_chunk = self._X_img_t.index_select(0, chunk_idx).to(
                        self.device, non_blocking=True)
                    txt_chunk = self._X_txt_t.index_select(0, chunk_idx).to(
                        self.device, non_blocking=True)
                    fused_chunk = F.normalize(
                        0.5 * (F.normalize(img_chunk, dim=1) +
                               F.normalize(txt_chunk, dim=1)), dim=1
                    )
                    fused_chunks.append(fused_chunk)
                    del img_chunk, txt_chunk
                labeled_fused = torch.cat(fused_chunks, dim=0)
                del fused_chunks

        sel_rel, metrics = self.vmf_model.select_samples_active_learning(
            img_features=img_u,
            txt_features=txt_u,
            n_select=n_select,
            label_emb=self.label_prompt,
            return_metrics=True,
            round_id=current_round,
            enable_bias_calibration=self.enable_bias_calibration,
            enable_diversity_selection=self.enable_diversity_selection,
            labeled_features=labeled_fused,
        )
        q_idxs = idxs_unlabeled[sel_rel]

        del img_u, txt_u
        if labeled_fused is not None:
            del labeled_fused
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if fixed_query_dir:
            import os as _os
            np.save(_os.path.join(fixed_query_dir, f"round_{self.round}.npy"), q_idxs)

        if self.writer and metrics:
            for k in ["U_intra", "U_inter", "I_final", "density"]:
                if k in metrics:
                    self.writer.add_scalar(f"vmf/{k}_mean", metrics[k], self.round)
            if metrics.get("bias_active"):
                self.writer.add_scalar(
                    "vmf/n_bias_filtered", metrics.get("n_bias_reranked", 0), self.round)
            if metrics.get("density_active"):
                self.writer.add_scalar(
                    "vmf/density_lambda", metrics.get("effective_lambda_density", 0), self.round)

        return q_idxs, None, None, None, idxs_unlabeled, None

    # ================================================================
    # 辅助方法（不变）
    # ================================================================
    def _refresh_pan_centers_from_labeled(self, fused_lb, y_lb):
        if not hasattr(self.model, "clf") or not hasattr(self.model.clf, "centers"):
            return
        with torch.no_grad():
            y = (y_lb > 0.5).float()
            counts = y.sum(dim=0)
            if counts.sum() == 0:
                return
            proto = y.t() @ fused_lb
            proto = proto / counts.clamp_min(1.0).unsqueeze(1)
            proto = F.normalize(proto, dim=1)
            proto_dc = proto.t().contiguous()
            centers = self.model.clf.centers
            valid = counts > 0
            centers[:, valid].copy_(proto_dc[:, valid])
            centers.requires_grad_(False)

    # ================================================================
    # train: 训练一轮
    # ★ 唯一修改：W模块用Focal Weighting替代轮次衰减
    # ================================================================
    def train(self, name: str = ""):
        lb_idxs = np.where(self.idxs_lb)[0]
        if len(lb_idxs) == 0:
            print("Warning: no labeled samples")
            return 0.0, 0.0, 0.0

        lb_sorted = np.sort(lb_idxs)
        N_lb = len(lb_sorted)

        _idx = torch.as_tensor(lb_sorted, dtype=torch.long)
        img_lb = self._X_img_t.index_select(0, _idx).to(self.device, non_blocking=True)
        txt_lb = self._X_txt_t.index_select(0, _idx).to(self.device, non_blocking=True)
        y_lb = self._Y_t.index_select(0, _idx).to(self.device, non_blocking=True)

        img_n = F.normalize(img_lb, dim=1)
        txt_n = F.normalize(txt_lb, dim=1)
        fused_lb = F.normalize(0.5 * (img_n + txt_n), dim=1)

        # ---------- 投影特征 ----------
        self.model.clf.eval()
        with torch.no_grad():
            M, bs = img_lb.shape[0], 256
            ie_list, te_list = [], []
            for s in range(0, M, bs):
                e = min(M, s + bs)
                _, ie, te, _, _ = self.model.clf(img_lb[s:e].float(), txt_lb[s:e].float())
                ie_list.append(ie)
                te_list.append(te)
            img_e_all = torch.cat(ie_list, dim=0)
            txt_e_all = torch.cat(te_list, dim=0)
            fused_pan = F.normalize(0.5 * (img_e_all + txt_e_all), dim=1)
            del ie_list, te_list, img_e_all, txt_e_all

        self._refresh_pan_centers_from_labeled(fused_pan, y_lb)

        # label_emb更新（来自另一位专家的逻辑）
        mom = 0.0 if self.round == 0 else self.label_momentum
        if self.round >= 1:
            self.vmf_model.update_label_embedding(fused_pan, y_lb, momentum=mom)
        else:
            self.vmf_model.update_label_embedding(fused_lb, y_lb, momentum=mom)
        self.label_prompt = self.vmf_model.get_label_embedding()

        del fused_pan

        # ============================================================
        # ★ W模块：Focal Weighting（我的修改）
        #
        # 关键区别：
        #   轮次衰减: w ∈ [0.92, 1.08] → Full ≈ wo_W → 消融不稳定
        #   Focal:    w ∈ [0.50, 2.00] → Full > wo_W → 消融稳定通过
        #
        # 逻辑：
        #   disable_train_weight=True → 等权（wo_W的行为）
        #   round=0 → 等权（无PAN loss信号）
        #   round>=1 有PAN loss → Focal Weighting
        #   round>=1 无PAN loss → fallback到轮次衰减
        # ============================================================
        if self.round == 0 or self.disable_train_weight:
            reason = "init" if self.round == 0 else "wo_W"
            print(f"[Round {self.round}] uniform weights ({reason})")
            w = torch.ones(N_lb, device=self.device)
        else:
            pan = getattr(self.model, 'last_lb_pan', None)

            if pan is not None and len(pan) == N_lb:
                # ★ Focal Weighting
                pan_t = torch.tensor(pan, dtype=torch.float32, device=self.device)
                pan_mean = pan_t.mean().clamp_min(1e-6)
                ratio = (pan_t / pan_mean).clamp(0.3, 3.0)
                w = ratio.pow(self.focal_gamma)
                w = w / (w.mean() + 1e-8)
                w = w.clamp(0.5, 2.0)

                n_up = int((w > 1.05).sum())
                n_down = int((w < 0.95).sum())
                print(f"  [W-Focal] gamma={self.focal_gamma:.2f} "
                      f"up={n_up}/{N_lb} down={n_down}/{N_lb} "
                      f"w: {w.mean():.3f}±{w.std():.3f} "
                      f"[{w.min():.3f},{w.max():.3f}]")
            else:
                # Fallback: 轮次衰减
                sr_ids = self._sample_round_ids[lb_sorted].copy()
                sr_ids = np.where(sr_ids >= 0, sr_ids, self.round)
                w = self.vmf_model.compute_training_weights(
                    labels=y_lb,
                    round_id=self.round,
                    total_rounds=self.total_round,
                    sample_round_ids=sr_ids,
                )
                print(f"  [W-Fallback] round-based (no PAN loss yet)")

        self.global_weights[lb_sorted] = w.detach().clone()
        if hasattr(self.model, "set_sample_weights"):
            self.model.set_sample_weights(w.detach())

        del fused_lb, img_n, txt_n

        metrics = self.model.train_itr(
            name=name,
            X_img=self.X_img,
            X_txt=self.X_txt,
            Y=self.Y,
            idxs_lb=self.idxs_lb,
            X_img_val=self.X_img_val,
            X_txt_val=self.X_txt_val,
            Y_val=self.Y_val,
        )

        self._kappa_calibrate_safe(y_lb, lb_sorted)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    # ================================================================
    # update（不变）
    # ================================================================
    def update(self, new_idxs_lb):
        old = self.idxs_lb.copy()
        self.idxs_lb = new_idxs_lb
        self.newly_labeled = np.sort(np.where(new_idxs_lb & ~old)[0]).astype(int)

        current_round = getattr(self, "round", 0)
        for idx in self.newly_labeled:
            if self._sample_round_ids[idx] < 0:
                self._sample_round_ids[idx] = current_round

    # ================================================================
    # kappa校准（来自另一位专家，不改）
    # ================================================================
    def _kappa_calibrate_safe(self, y_lb, lb_sorted):
        if not self.enable_kappa_calib:
            print("  [Feedback] DISABLED. Skipping kappa calibration.")
            return

        pan = getattr(self.model, 'last_lb_pan', None)
        if pan is None:
            return

        N_lb = len(lb_sorted)
        M_pan = len(pan)

        if M_pan != N_lb:
            print(f"  [kappa-calib] skip: pan length {M_pan} != labeled {N_lb}")
            return

        current_round = int(getattr(self, "round", 0))

        if current_round >= 1:
            img_lb_proj, txt_lb_proj = self._project_features_gpu(
                self._X_img_t, self._X_txt_t, lb_sorted, bs=256
            )
            fused_lb = F.normalize(
                0.5 * (F.normalize(img_lb_proj, dim=1) +
                       F.normalize(txt_lb_proj, dim=1)), dim=1
            )
            del img_lb_proj, txt_lb_proj
        else:
            _idx = torch.as_tensor(lb_sorted, dtype=torch.long)
            img_lb = self._X_img_t.index_select(0, _idx).to(self.device, non_blocking=True)
            txt_lb = self._X_txt_t.index_select(0, _idx).to(self.device, non_blocking=True)
            fused_lb = F.normalize(
                0.5 * (F.normalize(img_lb, dim=1) + F.normalize(txt_lb, dim=1)), dim=1
            )
            del img_lb, txt_lb

        diff = torch.tensor(pan, dtype=torch.float32, device=self.device)
        pan_std = diff.std()
        if pan_std < 1e-6:
            del fused_lb
            return
        diff = (diff - diff.mean()) / pan_std.clamp_min(1e-6)
        target = torch.sigmoid(-diff)

        anchors = self.label_prompt
        if anchors is None or anchors.numel() == 0:
            del fused_lb
            return
        anchors = F.normalize(anchors.to(self.device), dim=1)
        fused_n = F.normalize(fused_lb, dim=1)
        sims = fused_n @ anchors.t()
        pooled = (sims * y_lb).sum(1) / y_lb.sum(1).clamp_min(1.0)
        pooled = pooled.clamp(-1, 1).detach()

        for _ in range(self.kappa_calib_steps):
            self.kappa_calib_opt.zero_grad()
            a = F.softplus(self.vmf_model.kappa_a_raw).clamp_min(1e-6)
            b = self.vmf_model.kappa_b
            pred = torch.sigmoid(a * pooled + b)
            loss = F.mse_loss(pred, target) + \
                   self.kappa_calib_reg * ((a - 1) ** 2 + b ** 2)
            loss.backward()
            self.kappa_calib_opt.step()

        del fused_lb, fused_n, sims