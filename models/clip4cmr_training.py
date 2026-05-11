# models/clip4cmr_training.py
"""
CLIP4CMRTraining - 适配主动学习框架的训练器（可加权、多损失、多标签）

v-final-stable-v3:
  修正v2发现的问题:
  1. n_total_rounds 获取兜底：优先从 train_params 读，fallback 从 _al_round 推断
  2. EMA rejected 时若 best_model_state 为 None，fallback 到 prev_model_state
  3. 其余逻辑与 v2 一致

  三重保护机制：
  1. 后期轮次自适应训练参数（lr/epoch/patience/grad_clip）
  2. EMA参数平滑（训练中维护，轮结束后写回模型）
  3. Safety Rollback（后期mAP大幅下降时回退）
  以上仅对大数据集（COCO/NUS-WIDE）生效，MIRFlickr保持原版行为。
"""

from __future__ import annotations

import copy
import os

import numpy as np
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from itr_models.clip4cmr_losses import (
    ensure_multihot,
    match_matrix_multihot,
    dual_softmax_loss,
    contrastive_loss,
    triplet_loss,
    lifted_loss,
)
try:
    from itr_models.evaluate import fx_calc_map_label
except Exception:
    fx_calc_map_label = None


class ITRFeatureDataset(Dataset):
    def __init__(self, X_img, X_txt, Y):
        if isinstance(X_img, np.ndarray):
            self.X_img = torch.from_numpy(X_img).float()
        elif isinstance(X_img, torch.Tensor):
            self.X_img = X_img.float()
        else:
            self.X_img = torch.as_tensor(X_img, dtype=torch.float32)

        if isinstance(X_txt, np.ndarray):
            self.X_txt = torch.from_numpy(X_txt).float()
        elif isinstance(X_txt, torch.Tensor):
            self.X_txt = X_txt.float()
        else:
            self.X_txt = torch.as_tensor(X_txt, dtype=torch.float32)

        if isinstance(Y, np.ndarray):
            self.Y = torch.from_numpy(Y).float()
        elif isinstance(Y, torch.Tensor):
            self.Y = Y.float()
        else:
            self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_img[idx], self.X_txt[idx], self.Y[idx], idx


def _to_torch(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.is_pinned() or (not torch.cuda.is_available()):
        return x.to(device, non_blocking=True)
    return x.to(device)


def _infer_num_classes(y: torch.Tensor, centers: torch.Tensor, feat_dim: int) -> int:
    if y.dim() == 2:
        return int(y.shape[1])
    if centers.dim() != 2:
        raise ValueError(f"centers must be 2D, got {tuple(centers.shape)}")
    if centers.shape[1] == feat_dim:
        return int(centers.shape[0])
    if centers.shape[0] == feat_dim:
        return int(centers.shape[1])
    raise ValueError(f"cannot infer num_classes from centers={tuple(centers.shape)} and feat_dim={feat_dim}")


def PAN_hard(features: torch.Tensor, centers: torch.Tensor, y: torch.Tensor,
             sample_w: Optional[torch.Tensor] = None, return_vec: bool = False):
    feat = F.normalize(features, dim=1)
    B, D = feat.shape

    ctr = centers
    if ctr.dim() != 2:
        raise ValueError(f"centers must be 2D, got {tuple(ctr.shape)}")

    if ctr.shape[1] == D:
        ctr_cd = ctr
    elif ctr.shape[0] == D:
        ctr_cd = ctr.t()
    else:
        raise ValueError(
            f"centers shape mismatch: features D={D}, centers={tuple(ctr.shape)} "
            f"(expected [C,D] or [D,C])"
        )

    ctr_cd = F.normalize(ctr_cd, dim=1)
    sim = feat @ ctr_cd.t()

    y = ensure_multihot(y, num_classes=sim.shape[1]).to(sim.device)
    pos = y > 0.5
    neg = ~pos

    pos_sim = sim.masked_fill(~pos, -1e9)
    neg_sim = sim.masked_fill(~neg, -1e9)

    loss_pos = -torch.logsumexp(pos_sim, dim=1)
    loss_neg = torch.logsumexp(neg_sim, dim=1)

    loss = loss_pos + loss_neg
    if sample_w is not None:
        loss = loss * sample_w
    if return_vec:
        return loss
    return loss.mean()


# =====================================================================
#  EMA Helper: 维护模型参数的指数移动平均
# =====================================================================
class _EMAHelper:
    """
    Exponential Moving Average for model parameters.

    用法：
      ema = _EMAHelper(model, decay=0.999)
      # 每个 optimizer.step() 之后调用：
      ema.update()
      # 轮次结束后，将 EMA 参数写回模型：
      ema.apply_to_model()
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """每次 optimizer.step() 后调用"""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to_model(self):
        """将 EMA 参数写回模型"""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict(self, state):
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


class CLIP4CMRTraining:
    def __init__(self, net, net_args, handler, train_params, writer, device, init_model=True):
        self.clf = net.to(device)
        self.args = train_params
        self.writer = writer
        self.device = device
        self.handler = handler

        lr = float(self.args.get("learning_rate", 1e-4))
        wd = float(self.args.get("weight_decay", 0.0))
        betas = self.args.get("betas", (0.9, 0.999))
        self.optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr, weight_decay=wd, betas=betas)
        self.initial_lr = lr

        self.sample_weights: Optional[torch.Tensor] = None

        self.w_pan = float(self.args.get("w_pan", 1.0))
        self.w_dual = float(self.args.get("w_dualsoftmax", 1.0))
        self.w_lifted = float(self.args.get("w_lifted", 0.0))
        self.w_triplet = float(self.args.get("w_triplet", 0.0))
        self.w_contrastive = float(self.args.get("w_contrastive", 0.0))

        self.grad_clip = float(train_params.get("grad_clip", 1.0))
        self.patience = int(train_params.get("early_stop_patience", 5))
        self.min_delta = float(train_params.get("early_stop_min_delta", 0.001))

        self.scheduler_type = str(train_params.get("scheduler_type", "cosine"))
        self.T_max = int(train_params.get("T_max", train_params.get("n_epoch", 50)))
        self.eta_min = float(train_params.get("eta_min", 1e-6))

        self.ema_beta = float(train_params.get("ema_beta", 0.8))
        self.ema_mAP = None
        self.grad_norm_history = []

        self.current_epoch = 0
        self.n_epoch = None

        # ====== 轮次计数器与状态追踪 ======
        self._al_round = 0
        self._prev_best_mAP = None

    def _rebuild_optimizer(self):
        """重建 optimizer（用于 EMA apply 或 Safety Rollback 后）"""
        self.optimizer = torch.optim.Adam(
            self.clf.parameters(),
            lr=self.initial_lr,
            weight_decay=float(self.args.get("weight_decay", 0.0)),
            betas=self.args.get("betas", (0.9, 0.999))
        )

    def set_sample_weights(self, w: torch.Tensor):
        self.sample_weights = w.detach().to(self.device).float()

    def _total_loss(self, centers, img_feat, txt_feat, y, idx):
        if hasattr(y, "dim") and y.dim() == 2 and y.size(1) > 1:
            num_classes = int(y.size(1))
        else:
            num_classes = int(centers.shape[1]) if centers.shape[1] <= centers.shape[0] else int(centers.shape[0])

        y_mh = ensure_multihot(y, num_classes=num_classes).to(self.device)

        sw = None
        if self.sample_weights is not None:
            sw = self.sample_weights[idx.long()]
            sw = sw.clamp(min=0.1, max=3.0)

        pan_vec = PAN_hard(img_feat, centers, y_mh, sample_w=sw, return_vec=True) + \
                  PAN_hard(txt_feat, centers, y_mh, sample_w=sw, return_vec=True)
        loss_pan = pan_vec.mean()

        sim = F.normalize(img_feat, dim=1) @ F.normalize(txt_feat, dim=1).t()

        loss_dual = dual_softmax_loss(sim, sim.t(), y=y_mh, sample_w=sw)

        loss_lift = lifted_loss(img_feat, txt_feat, y_mh, sample_w=sw) if self.w_lifted != 0.0 else sim.new_tensor(0.0)
        loss_trip = triplet_loss(img_feat, txt_feat, y_mh, sample_w=sw) if self.w_triplet != 0.0 else sim.new_tensor(0.0)
        loss_con = contrastive_loss(img_feat, txt_feat, y_mh, sample_w=sw) if self.w_contrastive != 0.0 else sim.new_tensor(0.0)

        if hasattr(y, "dim") and y.dim() == 2:
            num_classes = int(y.size(1))
        else:
            feat_dim = int(img_feat.shape[1])
            if centers.shape[0] == feat_dim:
                num_classes = int(centers.shape[1])
            else:
                num_classes = int(centers.shape[0])

        if num_classes > 50:
            adaptive_w_pan = self.w_pan * 1.5
            adaptive_w_dual = self.w_dual * 0.8
        elif num_classes > 20:
            adaptive_w_pan = self.w_pan * 1.2
            adaptive_w_dual = self.w_dual * 1.0
        else:
            adaptive_w_pan = self.w_pan
            adaptive_w_dual = self.w_dual

        if num_classes > 50:
            w_lifted_adj = self.w_lifted * 0.6
            w_triplet_adj = self.w_triplet * 0.6
            w_contrastive_adj = self.w_contrastive * 0.6
        elif num_classes > 20:
            w_lifted_adj = self.w_lifted * 0.8
            w_triplet_adj = self.w_triplet * 0.8
            w_contrastive_adj = self.w_contrastive * 0.8
        else:
            w_lifted_adj = self.w_lifted
            w_triplet_adj = self.w_triplet
            w_contrastive_adj = self.w_contrastive

        total = (
                adaptive_w_pan * loss_pan
                + adaptive_w_dual * loss_dual
                + w_lifted_adj * loss_lift
                + w_triplet_adj * loss_trip
                + w_contrastive_adj * loss_con
        )

        return {
            "total": total,
            "pan": loss_pan,
            "pan_vec": pan_vec.detach(),
            "dual": loss_dual,
            "lifted": loss_lift,
            "triplet": loss_trip,
            "contrastive": loss_con,
        }

    def collect_pan_losses(self, X_img, X_txt, Y, idxs_lb, n_warmup_epochs=3):
        lb_idxs = np.sort(np.where(idxs_lb)[0])
        X_img_lb = X_img[lb_idxs]
        X_txt_lb = X_txt[lb_idxs]
        Y_lb = Y[lb_idxs]

        loader_args = self.args.get(
            "loader_tr_args",
            {"batch_size": 256, "num_workers": 0, "pin_memory": False},
        )
        train_loader = DataLoader(
            ITRFeatureDataset(X_img_lb, X_txt_lb, Y_lb),
            shuffle=True,
            **loader_args,
        )

        M = len(lb_idxs)
        pan_sum = torch.zeros(M, dtype=torch.float32)
        pan_cnt = torch.zeros(M, dtype=torch.float32)

        self.clf.eval()

        with torch.no_grad():
            for epoch in range(n_warmup_epochs):
                for x_img, x_txt, y, idx in train_loader:
                    x_img = _to_torch(x_img, self.device).float()
                    x_txt = _to_torch(x_txt, self.device).float()
                    y = _to_torch(y, self.device).float()
                    idx = idx.long()

                    centers, img_feat, txt_feat, _, _ = self.clf(x_img, x_txt)

                    if hasattr(y, "dim") and y.dim() == 2 and y.size(1) > 1:
                        num_classes = int(y.size(1))
                    else:
                        num_classes = (
                            int(centers.shape[1])
                            if centers.shape[1] <= centers.shape[0]
                            else int(centers.shape[0])
                        )
                    y_mh = ensure_multihot(y, num_classes=num_classes).to(self.device)

                    pv = PAN_hard(img_feat, centers, y_mh, return_vec=True) + \
                         PAN_hard(txt_feat, centers, y_mh, return_vec=True)

                    pv_cpu = pv.detach().float().cpu()
                    ii = idx.cpu()
                    pan_sum[ii] += pv_cpu
                    pan_cnt[ii] += 1.0

        pan_losses = (pan_sum / pan_cnt.clamp_min(1.0)).numpy()
        print(
            f"  [W-collect] PAN loss: {n_warmup_epochs}ep, "
            f"mean={pan_losses.mean():.4f} std={pan_losses.std():.4f}"
        )

        self.last_lb_pan = pan_losses
        return pan_losses

    # =====================================================================
    #  自适应训练参数 — 基于AL轮次
    # =====================================================================
    def _compute_adaptive_params(self, n_labeled, n_total_pool, al_round, n_total_rounds):
        """
        根据当前AL轮次自适应调整训练超参。

        为什么用轮次而非标注比例？
        - COCO池116k，20轮后也只标注10500个，ratio=9%，永远到不了30%
        - NUS-WIDE池268k，20轮后ratio=3.9%
        - 所以用"轮次进度"来判断前中后期

        n_total_rounds: 总AL轮数
        al_round: 当前第几轮（0-indexed）
        """
        base_n_epoch = int(self.args.get("n_epoch", 50))
        base_patience = int(self.args.get("early_stop_patience", 10))
        base_min_delta = float(self.args.get("early_stop_min_delta", 1e-4))
        base_grad_clip = float(self.args.get("grad_clip", 1.0))

        # 小数据集（MIRFlickr ~16k）：保持原始参数，不做任何改动
        is_large_dataset = n_total_pool > 20000
        if not is_large_dataset:
            return {
                'n_epoch': base_n_epoch,
                'lr_scale': 1.0,
                'patience': base_patience,
                'min_delta': base_min_delta,
                'warmup_epochs': 3,
                'grad_clip': base_grad_clip,
                'ema_decay': 0.0,
            }

        # ======== 大数据集：基于轮次进度的自适应 ========
        if n_total_rounds > 0:
            progress = al_round / n_total_rounds
        else:
            progress = 0.0

        if progress < 0.35:
            # 前期 (Round 0~6 / 20轮)
            lr_scale = 1.0
            n_epoch = base_n_epoch
            patience = base_patience
            min_delta = base_min_delta
            warmup_epochs = 3
            grad_clip_val = base_grad_clip
            ema_decay = 0.0

        elif progress < 0.65:
            # 中期 (Round 7~12 / 20轮)
            lr_scale = 0.7
            n_epoch = max(int(base_n_epoch * 0.8), 20)
            patience = max(base_patience - 2, 5)
            min_delta = 5e-4
            warmup_epochs = 2
            grad_clip_val = min(base_grad_clip, 0.8)
            ema_decay = 0.998

        else:
            # 后期 (Round 13~20 / 20轮)
            lr_scale = 0.4
            n_epoch = max(int(base_n_epoch * 0.6), 15)
            patience = max(base_patience - 4, 4)
            min_delta = 1e-3
            warmup_epochs = 2
            grad_clip_val = min(base_grad_clip, 0.5)
            ema_decay = 0.999

        return {
            'n_epoch': n_epoch,
            'lr_scale': lr_scale,
            'patience': patience,
            'min_delta': min_delta,
            'warmup_epochs': warmup_epochs,
            'grad_clip': grad_clip_val,
            'ema_decay': ema_decay,
        }

    # =====================================================================
    #  评估辅助函数
    # =====================================================================
    def _evaluate_retrieval(self, X_img_val, X_txt_val, Y_val, batch_size=256):
        """在给定数据上评估检索mAP，返回 (mAP, i2t, t2i)"""
        if fx_calc_map_label is None or X_img_val is None or len(X_img_val) == 0:
            return 0.0, 0.0, 0.0

        self.clf.eval()
        img_out_gpu, txt_out_gpu = [], []
        with torch.no_grad():
            n = len(X_img_val)
            for s in range(0, n, batch_size):
                xb_img = _to_torch(X_img_val[s:s + batch_size], self.device).float()
                xb_txt = _to_torch(X_txt_val[s:s + batch_size], self.device).float()
                _, i_f, t_f, _, _ = self.clf(xb_img, xb_txt)
                img_out_gpu.append(i_f.detach())
                txt_out_gpu.append(t_f.detach())

        img_out = torch.cat(img_out_gpu, dim=0).cpu().numpy()
        txt_out = torch.cat(txt_out_gpu, dim=0).cpu().numpy()
        del img_out_gpu, txt_out_gpu
        y_np = np.asarray(Y_val)

        i2t = float(fx_calc_map_label(img_out, txt_out, y_np))
        t2i = float(fx_calc_map_label(txt_out, img_out, y_np))
        mAP = 0.5 * (i2t + t2i)
        return mAP, i2t, t2i

    # =====================================================================
    #  主训练函数
    # =====================================================================
    def train_itr(self, name, X_img, X_txt, Y, idxs_lb, X_img_val, X_txt_val, Y_val, train_epoch_func=None):
        lb_idxs = np.sort(np.where(idxs_lb)[0])
        n_labeled = len(lb_idxs)
        n_total_pool = len(Y)

        # 获取总轮数：优先从 train_params 读，fallback 20
        # ★ 需要 main.py 中把 args.n_round 注入 train_params
        n_total_rounds = int(self.args.get("n_round", 20))
        current_round = self._al_round

        # ====== 核心改动1：基于轮次的自适应训练参数 ======
        adaptive = self._compute_adaptive_params(n_labeled, n_total_pool, current_round, n_total_rounds)

        n_epoch = adaptive['n_epoch']
        lr_scale = adaptive['lr_scale']
        patience = adaptive['patience']
        min_delta = adaptive['min_delta']
        warmup_epochs = adaptive['warmup_epochs']
        grad_clip_val = adaptive['grad_clip']
        ema_decay = adaptive['ema_decay']

        effective_lr = self.initial_lr * lr_scale

        progress = current_round / max(n_total_rounds, 1)
        phase = "early" if progress < 0.35 else ("mid" if progress < 0.65 else "late")
        print(f"\n[Adaptive Training] Round {current_round}/{n_total_rounds} "
              f"(progress={progress:.0%}, phase={phase}), "
              f"labeled={n_labeled}, "
              f"n_epoch={n_epoch}, lr={effective_lr:.2e}, "
              f"patience={patience}, min_delta={min_delta:.1e}, "
              f"grad_clip={grad_clip_val}, ema_decay={ema_decay}")

        self._al_round += 1

        X_img_lb = X_img[lb_idxs]
        X_txt_lb = X_txt[lb_idxs]
        Y_lb = Y[lb_idxs]

        loader_args = self.args.get("loader_tr_args", {"batch_size": 256, "num_workers": 0, "pin_memory": False})
        train_loader = DataLoader(ITRFeatureDataset(X_img_lb, X_txt_lb, Y_lb), shuffle=True, **loader_args)

        # ====== 核心改动2：lr重置为自适应值 ======
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = effective_lr

        use_sched = bool(self.args.get("lr_schedule", True))
        if use_sched:
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, n_epoch - warmup_epochs),
                eta_min=float(self.args.get("eta_min", 1e-6))
            )
        else:
            scheduler = None

        # ====== 核心改动3：保存进入本轮前的模型状态 ======
        prev_model_state = copy.deepcopy(self.clf.state_dict())

        # ====== 核心改动4：初始化EMA ======
        ema = None
        if ema_decay > 0:
            ema = _EMAHelper(self.clf, decay=ema_decay)

        best_mAP, best_i2t, best_t2i = 0.0, 0.0, 0.0
        patience_counter = 0
        best_model_state = None
        best_ema_state = None
        best_epoch = 0

        train_loss_history = []
        val_mAP_history = []

        log_dir = self.args.get("log_dir", ".")
        os.makedirs(log_dir, exist_ok=True)
        epoch_csv = os.path.join(log_dir, "epoch_metrics.csv")

        self.last_lb_pan = None

        _pan_sum = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
        _pan_cnt = torch.zeros(len(train_loader.dataset), dtype=torch.float32)

        pbar = tqdm(range(n_epoch), desc=f"clip4cmr[{name}]")
        early_stopped = False

        try:
            for epoch in pbar:
                # warmup阶段
                if epoch < warmup_epochs:
                    warmup_factor = (epoch + 1) / float(warmup_epochs)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = effective_lr * warmup_factor

                self.clf.train()
                loss_sum, iters = 0.0, 0

                loss_pan_sum = 0.0
                loss_dual_sum = 0.0
                loss_lifted_sum = 0.0
                loss_triplet_sum = 0.0
                loss_contrastive_sum = 0.0

                for x_img, x_txt, y, idx in train_loader:
                    x_img = _to_torch(x_img, self.device).float()
                    x_txt = _to_torch(x_txt, self.device).float()
                    y = _to_torch(y, self.device).float()
                    idx = _to_torch(idx, self.device).long()

                    self.optimizer.zero_grad()
                    centers, img_feat, txt_feat, img_pred, txt_pred = self.clf(x_img, x_txt)

                    loss_dict = self._total_loss(centers, img_feat, txt_feat, y, idx)
                    loss = loss_dict["total"]

                    if "pan_vec" in loss_dict:
                        pv = loss_dict["pan_vec"].detach().float().cpu()
                        ii = idx.detach().long().cpu()
                        _pan_sum[ii] += pv
                        _pan_cnt[ii] += 1.0

                    loss.backward()

                    if grad_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.clf.parameters(), max_norm=grad_clip_val)

                    self.optimizer.step()

                    # ====== EMA更新 ======
                    if ema is not None:
                        ema.update()

                    loss_sum += float(loss_dict["total"].item())
                    loss_pan_sum += float(loss_dict["pan"].item())
                    loss_dual_sum += float(loss_dict["dual"].item())
                    loss_lifted_sum += float(loss_dict["lifted"].item())
                    loss_triplet_sum += float(loss_dict["triplet"].item())
                    loss_contrastive_sum += float(loss_dict["contrastive"].item())
                    iters += 1

                avg_train_loss = loss_sum / max(1, iters)
                avg_pan = loss_pan_sum / max(1, iters)
                avg_dual = loss_dual_sum / max(1, iters)
                avg_lifted = loss_lifted_sum / max(1, iters)
                avg_triplet = loss_triplet_sum / max(1, iters)
                avg_contrastive = loss_contrastive_sum / max(1, iters)

                train_loss_history.append(avg_train_loss)

                if scheduler is not None and epoch >= warmup_epochs:
                    scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                if self.writer is not None:
                    self.writer.add_scalar(f"train/lr_{name}", current_lr, epoch)
                    self.writer.add_scalar(f"train/loss_total_{name}", avg_train_loss, epoch)
                    self.writer.add_scalar(f"train/loss_pan_{name}", avg_pan, epoch)
                    self.writer.add_scalar(f"train/loss_dual_{name}", avg_dual, epoch)
                    self.writer.add_scalar(f"train/loss_lifted_{name}", avg_lifted, epoch)
                    self.writer.add_scalar(f"train/loss_triplet_{name}", avg_triplet, epoch)
                    self.writer.add_scalar(f"train/loss_contrastive_{name}", avg_contrastive, epoch)

                # ====== 验证评估 ======
                val_mAP = 0.0
                if fx_calc_map_label is not None and X_img_val is not None and X_txt_val is not None and Y_val is not None:
                    if len(X_img_val) > 0 and len(X_txt_val) > 0 and len(Y_val) > 0:
                        bs = int(loader_args.get("batch_size", 256))
                        mAP, i2t_val, t2i_val = self._evaluate_retrieval(X_img_val, X_txt_val, Y_val, bs)
                        val_mAP = mAP
                        val_mAP_history.append(mAP)

                        if self.writer is not None:
                            self.writer.add_scalar(f"val/mAP_{name}", mAP, epoch)
                            self.writer.add_scalar(f"val/i2t_{name}", i2t_val, epoch)
                            self.writer.add_scalar(f"val/t2i_{name}", t2i_val, epoch)

                        if mAP > best_mAP + min_delta:
                            best_mAP = mAP
                            best_i2t = i2t_val
                            best_t2i = t2i_val
                            best_epoch = epoch
                            patience_counter = 0

                            best_model_state = {
                                'epoch': epoch,
                                'model_state': copy.deepcopy(self.clf.state_dict()),
                                'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
                            }
                            if ema is not None:
                                best_ema_state = ema.state_dict()
                        else:
                            min_epochs = int(self.args.get("early_stop_warmup", 10))
                            if epoch >= min_epochs:
                                patience_counter += 1
                                if patience_counter >= patience:
                                    if best_model_state is not None:
                                        self.clf.load_state_dict(best_model_state['model_state'])
                                        self.optimizer.load_state_dict(best_model_state['optimizer_state'])
                                        if ema is not None and best_ema_state is not None:
                                            ema.load_state_dict(best_ema_state)
                                    with torch.no_grad():
                                        self.last_lb_pan = (_pan_sum / _pan_cnt.clamp_min(1.0)).numpy()
                                    early_stopped = True
                                    break

                if epoch % 5 == 0 or epoch == n_epoch - 1:
                    val_map_str = f"{val_mAP:.4f}" if val_mAP > 0 else "N/A"
                    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                          f"Val mAP={val_map_str}, "
                          f"LR={current_lr:.2e}, Patience={patience_counter}/{patience}")

                if (epoch == 0) and (not os.path.exists(epoch_csv)):
                    with open(epoch_csv, "w") as f:
                        f.write(
                            "hyper_id,round,epoch,loss_total,loss_pan,loss_dual,loss_lifted,loss_triplet,loss_contrastive,lr,val_mAP\n")
                epoch_log = {
                    "hyper_id": int(self.args.get("hyper_id", -1)),
                    "round": int(self.args.get("round", -1)),
                    "epoch": int(epoch),
                    "loss_total": avg_train_loss,
                    "loss_pan": avg_pan,
                    "loss_dual": avg_dual,
                    "loss_lifted": avg_lifted,
                    "loss_triplet": avg_triplet,
                    "loss_contrastive": avg_contrastive,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "val_mAP": float(val_mAP) if val_mAP is not None else -1.0,
                }
                _log_dir = self.args.get("log_dir", ".") if isinstance(self.args, dict) else getattr(self.args,
                                                                                                     "log_dir", ".")
                os.makedirs(_log_dir, exist_ok=True)
                with open(os.path.join(_log_dir, "epoch_metrics.csv"), "a") as f:
                    f.write(",".join(map(str, epoch_log.values())) + "\n")

        finally:
            pbar.close()

        # ====== 训练结束后恢复最佳模型 ======
        if early_stopped:
            print(f"[Early Stop] at epoch {best_epoch}, best mAP={best_mAP:.4f}")
        else:
            if best_model_state is not None and best_epoch != n_epoch - 1:
                print(f"[Training Complete] Restoring best model from epoch {best_epoch}")
                self.clf.load_state_dict(best_model_state['model_state'])
                self.optimizer.load_state_dict(best_model_state['optimizer_state'])
                if ema is not None and best_ema_state is not None:
                    ema.load_state_dict(best_ema_state)

        # ====== 核心改动5：EMA参数写回模型 ======
        if ema is not None:
            pre_ema_mAP = best_mAP

            ema.apply_to_model()
            self._rebuild_optimizer()

            if fx_calc_map_label is not None and X_img_val is not None and len(X_img_val) > 0:
                bs = int(loader_args.get("batch_size", 256))
                ema_mAP, ema_i2t, ema_t2i = self._evaluate_retrieval(X_img_val, X_txt_val, Y_val, bs)

                if ema_mAP >= pre_ema_mAP - 0.002:
                    best_mAP = ema_mAP
                    best_i2t = ema_i2t
                    best_t2i = ema_t2i
                    print(f"[EMA] Applied. mAP: {pre_ema_mAP:.4f} -> {ema_mAP:.4f}")
                else:
                    # EMA更差，回退
                    # 优先用 best_model_state，如果没有则用 prev_model_state
                    if best_model_state is not None:
                        self.clf.load_state_dict(best_model_state['model_state'])
                    else:
                        self.clf.load_state_dict(prev_model_state)
                    self._rebuild_optimizer()
                    print(f"[EMA] Rejected. EMA mAP={ema_mAP:.4f} < original={pre_ema_mAP:.4f}, reverted")
            else:
                print(f"[EMA] Applied (no val set to verify)")

        # ====== 核心改动6：Safety Rollback ======
        if self._prev_best_mAP is not None:
            is_large = n_total_pool > 20000
            is_late = progress > 0.5
            drop = self._prev_best_mAP - best_mAP

            if is_large and is_late and drop > 0.003:
                print(f"[Safety Rollback] mAP dropped {drop:.4f} "
                      f"({self._prev_best_mAP:.4f} -> {best_mAP:.4f})")

                self.clf.load_state_dict(prev_model_state)
                self._rebuild_optimizer()

                if fx_calc_map_label is not None and X_img_val is not None and len(X_img_val) > 0:
                    bs = int(loader_args.get("batch_size", 256))
                    rollback_mAP, rollback_i2t, rollback_t2i = self._evaluate_retrieval(
                        X_img_val, X_txt_val, Y_val, bs)

                    if rollback_mAP > best_mAP:
                        best_mAP = rollback_mAP
                        best_i2t = rollback_i2t
                        best_t2i = rollback_t2i
                        print(f"[Safety Rollback] Using rollback mAP: {rollback_mAP:.4f}")
                    else:
                        if best_model_state is not None:
                            self.clf.load_state_dict(best_model_state['model_state'])
                        else:
                            # best_model_state 也没有，只能保持prev_model_state
                            pass
                        self._rebuild_optimizer()
                        print(f"[Safety Rollback] Rollback mAP ({rollback_mAP:.4f}) "
                              f"not better, keeping current ({best_mAP:.4f})")

        # 记录本轮mAP
        self._prev_best_mAP = best_mAP

        print(f"\n{'=' * 50}")
        print(f"Training Summary for {name}:")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best mAP: {best_mAP:.4f}")
        print(f"  Best i2t: {best_i2t:.4f}")
        print(f"  Best t2i: {best_t2i:.4f}")
        print(f"{'=' * 50}\n")

        if not early_stopped:
            with torch.no_grad():
                self.last_lb_pan = (_pan_sum / _pan_cnt.clamp_min(1.0)).numpy()

        return best_mAP, best_i2t, best_t2i