# itr_models/clip4cmr_losses316.py
import math
import torch
import torch.nn.functional as F


def ensure_multihot(y: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    """
    强制输出 multi-hot float [B,C]
    - 若 y 为 [B] long：转 one-hot
    - 若 y 为 [B,C]：float 化
    """
    if y.dim() == 1:
        if num_classes is None:
            num_classes = int(y.max().item()) + 1
        out = torch.zeros((y.size(0), num_classes), device=y.device, dtype=torch.float32)
        out.scatter_(1, y.long().view(-1, 1), 1.0)
        return out
    return y.float()


def match_matrix_multihot(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """
    multi-hot 匹配矩阵：
    正样本：标签交集 > 0
    y1: [B,C], y2: [B,C]
    return: [B,B] bool
    """
    y1 = y1.float()
    y2 = y2.float()
    overlap = y1 @ y2.t()
    return overlap > 0


# def dual_softmax_loss(sim_i2t: torch.Tensor, sim_t2i: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
#     """
#     Dual Softmax / 对称 InfoNCE
#     sim_i2t: [B,B] cosine similarity
#     """
#     sim_i2t = sim_i2t / temp
#     sim_t2i = sim_t2i / temp
#     B = sim_i2t.size(0)
#     target = torch.arange(B, device=sim_i2t.device)
#
#     loss_i2t = F.cross_entropy(sim_i2t, target)
#     loss_t2i = F.cross_entropy(sim_t2i, target)
#     return 0.5 * (loss_i2t + loss_t2i)

# 在 clip4cmr_losses316.py 中添加新函数并修改现有函数

def adaptive_temperature_dual_softmax_loss(
        sim_i2t: torch.Tensor,
        sim_t2i: torch.Tensor,
        y: torch.Tensor,
        base_temp: float = 0.05,
        sample_w=None, # [新增参数]
        min_temp: float = 0.01,
        max_temp: float = 0.1
) -> torch.Tensor:
    """
    自适应温度的Dual Softmax损失
    根据样本难度动态调整温度参数
    """
    # 计算每个样本的难度（标签数量越少，难度越高）
    label_counts = y.sum(dim=1).float()  # [B]
    avg_label_count = label_counts.mean().item()

    # 归一化标签数量到 [0.5, 1.5] 范围
    norm_factor = label_counts / avg_label_count
    norm_factor = torch.clamp(norm_factor, 0.5, 1.5)

    # 为每个样本计算自适应温度：标签少的样本用更低温度（更尖锐）
    # adaptive_temp = base_temp * (2.0 - norm_factor)  # 标签越少，温度越低
    adaptive_temp = (base_temp * norm_factor).clamp(min_temp, max_temp)

    # 每行使用自己的温度
    sim_i2t_scaled = sim_i2t / adaptive_temp.unsqueeze(1)  # [B,B] / [B,1]
    sim_t2i_scaled = sim_t2i / adaptive_temp.unsqueeze(1)

    # 多标签匹配矩阵
    m = match_matrix_multihot(y, y)

    # 计算损失
    logp_i2t = F.log_softmax(sim_i2t_scaled, dim=1)
    logp_t2i = F.log_softmax(sim_t2i_scaled, dim=1)

    # loss_i2t = -torch.logsumexp(logp_i2t.masked_fill(~m, -1e9), dim=1).mean()
    # loss_t2i = -torch.logsumexp(logp_t2i.masked_fill(~m, -1e9), dim=1).mean()

    loss_vec_i2t = -torch.logsumexp(logp_i2t.masked_fill(~m, -1e9), dim=1)
    loss_vec_t2i = -torch.logsumexp(logp_t2i.masked_fill(~m, -1e9), dim=1)

    # [修改]：如果传入了权重，应用到向量上
    if sample_w is not None:
        loss_vec_i2t = loss_vec_i2t * sample_w
        loss_vec_t2i = loss_vec_t2i * sample_w

    return 0.5 * (loss_vec_i2t.mean() + loss_vec_t2i.mean())

    # return 0.5 * (loss_i2t + loss_t2i)


# 修改原来的dual_softmax_loss函数，调用新的自适应版本
# def dual_softmax_loss(sim_i2t: torch.Tensor, sim_t2i: torch.Tensor, y: torch.Tensor | None = None, temp: float = 0.05,sample_w=None):
#     if y is None:
#         B = sim_i2t.size(0)
#         target = torch.arange(B, device=sim_i2t.device)
#         return 0.5 * (F.cross_entropy(sim_i2t / temp, target) + F.cross_entropy(sim_t2i / temp, target))
#
#     # 使用自适应温度版本
#     return adaptive_temperature_dual_softmax_loss(sim_i2t, sim_t2i, y, base_temp=temp,sample_w=sample_w)

def dual_softmax_loss(sim: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    带有主动学习样本权重的 Dual Softmax Loss
    sim: [B, B] 的相似度矩阵
    weights: [B] 的主动学习样本权重 (如果传入)
    """
    # 提取对角线（正样本），加入极小值防 log(0)
    sim_i2t = F.softmax(sim, dim=1).diag() + 1e-8
    sim_t2i = F.softmax(sim, dim=0).diag() + 1e-8

    # 分别计算两个方向的 InfoNCE loss（不立刻求均值，保留为 [B] 的形状）
    loss_i2t = -torch.log(sim_i2t)
    loss_t2i = -torch.log(sim_t2i)

    # 计算每个样本的独立 loss: shape [B]
    loss_per_sample = (loss_i2t + loss_t2i) / 2.0

    # 【核心修正4】：精准注入样本权重进入计算图！
    if weights is not None:
        # weights 的 shape 应为 [B]，与 loss_per_sample 对应相乘
        loss_per_sample = loss_per_sample * weights.view(-1)

    return loss_per_sample.mean()

# def dual_softmax_loss(sim_i2t: torch.Tensor, sim_t2i: torch.Tensor, y: torch.Tensor | None = None, temp: float = 0.05):
#     sim_i2t = sim_i2t / temp
#     sim_t2i = sim_t2i / temp
#
#     if y is None:
#         B = sim_i2t.size(0)
#         target = torch.arange(B, device=sim_i2t.device)
#         return 0.5 * (F.cross_entropy(sim_i2t, target) + F.cross_entropy(sim_t2i, target))
#
#     # multi-hot positives
#     m = match_matrix_multihot(y, y)  # [B,B] bool
#
#     logp_i2t = F.log_softmax(sim_i2t, dim=1)
#     logp_t2i = F.log_softmax(sim_t2i, dim=1)
#
#     # 每个 anchor 的 loss = -log sum_{pos} p(pos|anchor)
#     loss_i2t = -torch.logsumexp(logp_i2t.masked_fill(~m, -1e9), dim=1).mean()
#     loss_t2i = -torch.logsumexp(logp_t2i.masked_fill(~m, -1e9), dim=1).mean()
#
#     return 0.5 * (loss_i2t + loss_t2i)



# def contrastive_loss(img: torch.Tensor, txt: torch.Tensor, y: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
#     """
#     跨模态 contrastive（multi-label）：
#     - 对每个 i，正样本集合 P(i)：match_matrix[i,:] 为 True
#     - 负样本集合 N(i)：否则
#     使用 pairwise hinge：max(0, margin - s_pos + s_neg)
#     """
#     img = F.normalize(img, dim=1)
#     txt = F.normalize(txt, dim=1)
#     sim = img @ txt.t()  # [B,B]
#     m = match_matrix_multihot(y, y)  # [B,B]
#
#     # 为稳定起见：每行取最难正样本（最小 sim）和最难负样本（最大 sim）
#     # 若某行没有正样本（极少），就跳过
#     B = sim.size(0)
#     losses = []
#     for i in range(B):
#         pos = sim[i][m[i]]
#         neg = sim[i][~m[i]]
#         if pos.numel() == 0 or neg.numel() == 0:
#             continue
#         s_pos = pos.min()
#         s_neg = neg.max()
#         losses.append(F.relu(margin - s_pos + s_neg))
#     if len(losses) == 0:
#         return torch.tensor(0.0, device=sim.device)
#     return torch.stack(losses).mean()
#
#
# def triplet_loss(img: torch.Tensor, txt: torch.Tensor, y: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
#     """
#     跨模态 triplet（multi-label）：
#     anchor=img_i, pos=txt_j (match), neg=txt_k (not match)
#     hardest pos/neg 版本
#     """
#     img = F.normalize(img, dim=1)
#     txt = F.normalize(txt, dim=1)
#     sim = img @ txt.t()  # [B,B]
#     m = match_matrix_multihot(y, y)
#
#     losses = []
#     B = sim.size(0)
#     for i in range(B):
#         pos = sim[i][m[i]]
#         neg = sim[i][~m[i]]
#         if pos.numel() == 0 or neg.numel() == 0:
#             continue
#         # hardest: pos 最小，neg 最大
#         s_pos = pos.min()
#         s_neg = neg.max()
#         losses.append(F.relu(margin - s_pos + s_neg))
#     if len(losses) == 0:
#         return torch.tensor(0.0, device=sim.device)
#     return torch.stack(losses).mean()
#
#
# def lifted_loss(img: torch.Tensor, txt: torch.Tensor, y: torch.Tensor,sample_w=None) -> torch.Tensor:
#     """
#     简化稳定版 lifted-structured（跨模态）：
#     对每个 anchor i：
#       L_i = log(1 + sum_{neg} exp(sim_neg - sim_pos_min))
#     其中 sim_pos_min 是该 anchor 的“最难正样本”
#     """
#     img = F.normalize(img, dim=1)
#     txt = F.normalize(txt, dim=1)
#     sim = img @ txt.t()
#     m = match_matrix_multihot(y, y)
#
#     losses = []
#     B = sim.size(0)
#     for i in range(B):
#         pos = sim[i][m[i]]
#         neg = sim[i][~m[i]]
#         if pos.numel() == 0 or neg.numel() == 0:
#             continue
#         s_pos = pos.min()
#         # log(1 + sum exp(neg - s_pos))
#         losses.append(torch.log1p(torch.exp(neg - s_pos).sum()))
#     if len(losses) == 0:
#         return torch.tensor(0.0, device=sim.device)
#     return torch.stack(losses).mean()

def contrastive_loss(img: torch.Tensor, txt: torch.Tensor, y: torch.Tensor, margin: float = 0.3,
                     sample_w=None) -> torch.Tensor:
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    sim = img @ txt.t()
    m = match_matrix_multihot(y, y)

    B = sim.size(0)
    losses = []
    for i in range(B):
        pos = sim[i][m[i]]
        neg = sim[i][~m[i]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        s_pos = pos.min()
        s_neg = neg.max()
        # 计算单个样本的 hinge loss
        raw_loss = F.relu(margin - s_pos + s_neg)
        # 应用样本权重
        if sample_w is not None:
            raw_loss = raw_loss * sample_w[i]
        losses.append(raw_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=sim.device)
    return torch.stack(losses).mean()


def triplet_loss(img: torch.Tensor, txt: torch.Tensor, y: torch.Tensor, margin: float = 0.2,
                 sample_w=None) -> torch.Tensor:
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    sim = img @ txt.t()
    m = match_matrix_multihot(y, y)

    losses = []
    B = sim.size(0)
    for i in range(B):
        pos = sim[i][m[i]]
        neg = sim[i][~m[i]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        s_pos = pos.min()
        s_neg = neg.max()
        # 计算单个样本的 triplet loss
        raw_loss = F.relu(margin - s_pos + s_neg)
        # 应用样本权重
        if sample_w is not None:
            raw_loss = raw_loss * sample_w[i]
        losses.append(raw_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=sim.device)
    return torch.stack(losses).mean()


def lifted_loss(img: torch.Tensor, txt: torch.Tensor, y: torch.Tensor, sample_w=None) -> torch.Tensor:
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    sim = img @ txt.t()
    m = match_matrix_multihot(y, y)

    losses = []
    B = sim.size(0)
    for i in range(B):
        pos = sim[i][m[i]]
        neg = sim[i][~m[i]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        s_pos = pos.min()
        # 计算单个样本的 lifted loss
        raw_loss = torch.log1p(torch.exp(neg - s_pos).sum())
        # 应用样本权重
        if sample_w is not None:
            raw_loss = raw_loss * sample_w[i]
        losses.append(raw_loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=sim.device)
    return torch.stack(losses).mean()


def modality_invariant_loss(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    """
    简化稳定版 modality-invariant：
    - 约束两个模态的均值/协方差接近（这里用均值+方差）
    """
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    mu_i, mu_t = img.mean(0), txt.mean(0)
    var_i, var_t = img.var(0, unbiased=False), txt.var(0, unbiased=False)
    return F.mse_loss(mu_i, mu_t) + F.mse_loss(var_i, var_t)


def proxy_anchor_loss(feature: torch.Tensor, y: torch.Tensor, proxies: torch.Tensor,
                      alpha: float = 32.0, beta: float = 32.0, margin: float = 0.1) -> torch.Tensor:
    """
    Proxy Anchor / PAN 的多标签版（稳定实现）
    feature: [B,D] normalized
    y: [B,C] multi-hot
    proxies: [C,D] normalized
    """
    feature = F.normalize(feature, dim=1)
    proxies = F.normalize(proxies, dim=1)
    y = y.float()

    sim = feature @ proxies.t()  # [B,C]

    # 正负 mask
    pos = y > 0
    neg = ~pos

    # 正项：log(1 + sum exp(-alpha*(s - margin)))
    # 负项：log(1 + sum exp(beta*(s + margin)))
    # 按类聚合更接近 PAN 原式
    loss_pos = torch.tensor(0.0, device=feature.device)
    loss_neg = torch.tensor(0.0, device=feature.device)

    # 避免空类
    for c in range(sim.size(1)):
        sp = sim[:, c][pos[:, c]]
        sn = sim[:, c][neg[:, c]]
        if sp.numel() > 0:
            loss_pos = loss_pos + torch.log1p(torch.exp(-alpha * (sp - margin)).sum())
        if sn.numel() > 0:
            loss_neg = loss_neg + torch.log1p(torch.exp(beta * (sn + margin)).sum())

    # 归一化
    return (loss_pos + loss_neg) / sim.size(1)
