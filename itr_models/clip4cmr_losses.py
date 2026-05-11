# itr_models/clip4cmr_losses.py
import math
import torch
import torch.nn.functional as F


def ensure_multihot(y: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    """强制输出 multi-hot float [B,C]"""
    if y.dim() == 1:
        if num_classes is None:
            num_classes = int(y.max().item()) + 1
        out = torch.zeros((y.size(0), num_classes), device=y.device, dtype=torch.float32)
        out.scatter_(1, y.long().view(-1, 1), 1.0)
        return out
    return y.float()


def match_matrix_multihot(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """
    multi-hot 匹配矩阵：正样本：标签交集 > 0
    y1: [B,C], y2: [B,C]  ->  return: [B,B] bool
    """
    y1 = y1.float()
    y2 = y2.float()
    overlap = y1 @ y2.t()
    return overlap > 0


def dual_softmax_loss(
    sim: torch.Tensor,
    sim_t: torch.Tensor = None,
    y: torch.Tensor = None,
    temp: float = 0.05,
    sample_w: torch.Tensor = None,
) -> torch.Tensor:
    """
    多标签版 Dual Softmax Loss（修复对角线单标签Bug）。

    原实现错误：
        sim_i2t = F.softmax(sim, dim=1).diag()   ← 只取对角线，假设 batch[i] 只和 batch[i] 匹配
        这在多标签数据集上把大量合法正样本对当做负样本惩罚，严重污染梯度。

    修复方案：
        使用 match_matrix_multihot 构建完整正样本矩阵 m [B,B]，
        loss_i2t[i] = -logsumexp( logp[i, j] for j in positives(i) )
        等价于：anchor i 至少检索到一个正样本的对数概率的负值。
        这与 mAP 评估指标的语义一致（任一正样本被检索到即有贡献）。

    参数:
        sim  : [B,B] 余弦相似度矩阵（i2t方向，未除温度）
        sim_t: [B,B] 余弦相似度矩阵（t2i方向），默认 sim.t()
        y    : [B,C] multi-hot 标签，用于构建正样本矩阵；
               若为 None 则退化为单标签对角线版本（仅用于调试）
        temp : 温度系数，默认 0.05
    """
    if sim_t is None:
        sim_t = sim.t()

    # ---- y 为 None 时退化为单标签版本（向后兼容，实际不应触发）----
    if y is None:
        B = sim.size(0)
        target = torch.arange(B, device=sim.device)
        return 0.5 * (
            F.cross_entropy(sim / temp, target) +
            F.cross_entropy(sim_t / temp, target)
        )

    # ---- 多标签正确实现 ----
    y = y.float()
    m = match_matrix_multihot(y, y)  # [B,B] bool，正样本对为 True

    # 温度缩放后做 log_softmax
    logp_i2t = F.log_softmax(sim / temp, dim=1)   # [B,B]
    logp_t2i = F.log_softmax(sim_t / temp, dim=1) # [B,B]

    # 每个 anchor i 的损失 = -log P(任一正样本 | anchor i)
    # = -logsumexp over j∈positives(i) of logp[i,j]
    # masked_fill: 非正样本位置置 -1e9（softmax后概率≈0，不参与logsumexp）
    loss_i2t = -torch.logsumexp(logp_i2t.masked_fill(~m, -1e9), dim=1)  # [B]
    loss_t2i = -torch.logsumexp(logp_t2i.masked_fill(~m, -1e9), dim=1)  # [B]

    # 对角线自匹配必须在正样本集合内（确保 m[i,i]=True 防止 loss=inf）
    # 当 batch 内有孤立样本（所有 m[i,:]=False）时给出保护
    has_pos = m.any(dim=1)
    loss_i2t = loss_i2t[has_pos]
    loss_t2i = loss_t2i[has_pos]

    if loss_i2t.numel() == 0:
        return sim.new_tensor(0.0)

    # ======== Fix-6 新增：应用样本权重 ========
    if sample_w is not None:
        sw_valid = sample_w[has_pos]
        loss_i2t = loss_i2t * sw_valid
        loss_t2i = loss_t2i * sw_valid
    # ======== Fix-6 结束 ========

    return 0.5 * (loss_i2t.mean() + loss_t2i.mean())


# def contrastive_loss(
#     img: torch.Tensor,
#     txt: torch.Tensor,
#     y: torch.Tensor,
#     margin: float = 0.3,
#     sample_w=None,
# ) -> torch.Tensor:
#     img = F.normalize(img, dim=1)
#     txt = F.normalize(txt, dim=1)
#     sim = img @ txt.t()
#     m = match_matrix_multihot(y, y)
#
#     B = sim.size(0)
#     losses = []
#     for i in range(B):
#         pos = sim[i][m[i]]
#         neg = sim[i][~m[i]]
#         if pos.numel() == 0 or neg.numel() == 0:
#             continue
#         s_pos = pos.min()
#         s_neg = neg.max()
#         raw_loss = F.relu(margin - s_pos + s_neg)
#         if sample_w is not None:
#             raw_loss = raw_loss * sample_w[i]
#         losses.append(raw_loss)
#
#     if len(losses) == 0:
#         return torch.tensor(0.0, device=sim.device)
#     return torch.stack(losses).mean()
#
#
# def triplet_loss(
#     img: torch.Tensor,
#     txt: torch.Tensor,
#     y: torch.Tensor,
#     margin: float = 0.2,
#     sample_w=None,
# ) -> torch.Tensor:
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
#         s_neg = neg.max()
#         raw_loss = F.relu(margin - s_pos + s_neg)
#         if sample_w is not None:
#             raw_loss = raw_loss * sample_w[i]
#         losses.append(raw_loss)
#
#     if len(losses) == 0:
#         return torch.tensor(0.0, device=sim.device)
#     return torch.stack(losses).mean()
#
#
# def lifted_loss(
#     img: torch.Tensor,
#     txt: torch.Tensor,
#     y: torch.Tensor,
#     sample_w=None,
# ) -> torch.Tensor:
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
#         raw_loss = torch.log1p(torch.exp(neg - s_pos).sum())
#         if sample_w is not None:
#             raw_loss = raw_loss * sample_w[i]
#         losses.append(raw_loss)
#
#     if len(losses) == 0:
#         return torch.tensor(0.0, device=sim.device)
#     return torch.stack(losses).mean()

def contrastive_loss(
    img: torch.Tensor,
    txt: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.3,
    sample_w=None,
) -> torch.Tensor:
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    sim = img @ txt.t()                          # [B, B]
    m = match_matrix_multihot(y, y)               # [B, B] bool

    has_pos = m.any(dim=1)
    has_neg = (~m).any(dim=1)
    valid = has_pos & has_neg

    if not valid.any():
        return torch.tensor(0.0, device=sim.device)

    # hardest positive: min sim among positives
    pos_sim = sim.masked_fill(~m, float("inf"))
    s_pos = pos_sim.min(dim=1).values             # [B]

    # hardest negative: max sim among negatives
    neg_sim = sim.masked_fill(m, float("-inf"))
    s_neg = neg_sim.max(dim=1).values             # [B]

    loss = F.relu(margin - s_pos + s_neg)
    loss = loss[valid]

    if sample_w is not None:
        loss = loss * sample_w[valid]

    return loss.mean()


def triplet_loss(
    img: torch.Tensor,
    txt: torch.Tensor,
    y: torch.Tensor,
    margin: float = 0.2,
    sample_w=None,
) -> torch.Tensor:
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    sim = img @ txt.t()
    m = match_matrix_multihot(y, y)

    has_pos = m.any(dim=1)
    has_neg = (~m).any(dim=1)
    valid = has_pos & has_neg

    if not valid.any():
        return torch.tensor(0.0, device=sim.device)

    pos_sim = sim.masked_fill(~m, float("inf"))
    s_pos = pos_sim.min(dim=1).values

    neg_sim = sim.masked_fill(m, float("-inf"))
    s_neg = neg_sim.max(dim=1).values

    loss = F.relu(margin - s_pos + s_neg)
    loss = loss[valid]

    if sample_w is not None:
        loss = loss * sample_w[valid]

    return loss.mean()


def lifted_loss(
    img: torch.Tensor,
    txt: torch.Tensor,
    y: torch.Tensor,
    sample_w=None,
) -> torch.Tensor:
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    sim = img @ txt.t()                           # [B, B]
    m = match_matrix_multihot(y, y)

    has_pos = m.any(dim=1)
    has_neg = (~m).any(dim=1)
    valid = has_pos & has_neg

    if not valid.any():
        return torch.tensor(0.0, device=sim.device)

    # hardest positive
    pos_sim = sim.masked_fill(~m, float("inf"))
    s_pos = pos_sim.min(dim=1).values             # [B]

    # lifted: log(1 + sum exp(neg - s_pos))
    # 先广播 s_pos: [B,1]
    diff = sim - s_pos.unsqueeze(1)               # [B, B]
    # 只对负样本求和
    diff_neg = diff.masked_fill(m, float("-inf"))
    # logsumexp over negatives -> log(sum exp(neg - s_pos))
    lse = torch.logsumexp(diff_neg, dim=1)        # [B]
    loss = torch.log1p(lse.exp())                 # log(1 + sum exp(...))
    # 上面等价于 log1p(exp(logsumexp(...)))
    # 更稳定的写法: softplus(logsumexp)
    loss = F.softplus(lse)

    loss = loss[valid]

    if sample_w is not None:
        loss = loss * sample_w[valid]

    return loss.mean()



def modality_invariant_loss(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    img = F.normalize(img, dim=1)
    txt = F.normalize(txt, dim=1)
    mu_i, mu_t = img.mean(0), txt.mean(0)
    var_i, var_t = img.var(0, unbiased=False), txt.var(0, unbiased=False)
    return F.mse_loss(mu_i, mu_t) + F.mse_loss(var_i, var_t)


def proxy_anchor_loss(
    feature: torch.Tensor,
    y: torch.Tensor,
    proxies: torch.Tensor,
    alpha: float = 32.0,
    beta: float = 32.0,
    margin: float = 0.1,
) -> torch.Tensor:
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
    pos = y > 0
    neg = ~pos

    loss_pos = torch.tensor(0.0, device=feature.device)
    loss_neg = torch.tensor(0.0, device=feature.device)

    for c in range(sim.size(1)):
        sp = sim[:, c][pos[:, c]]
        sn = sim[:, c][neg[:, c]]
        if sp.numel() > 0:
            loss_pos = loss_pos + torch.log1p(torch.exp(-alpha * (sp - margin)).sum())
        if sn.numel() > 0:
            loss_neg = loss_neg + torch.log1p(torch.exp(beta * (sn + margin)).sum())

    return (loss_pos + loss_neg) / sim.size(1)
