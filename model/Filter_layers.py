import torch
import torch.nn as nn
import torch.nn.functional as F

class mask_moe(nn.Module):
    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=12):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim

        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p
        self.seen= 0

    def cv2_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var(unbiased=False) / (x.float().mean() ** 2 + eps)
    def cross_entropy(self, x):   #
        eps = 1e-10
        if x.shape[0] == 1:
             return torch.tensor([0], device=x.device, dtype=x.dtype)
        ent = -(x * (x + 1e-9).log()).sum(dim=-1).mean()
        return  torch.relu(0.5 - ent)
    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            noise_stddev = ((self.softplus(raw_noise) + noise_epsilon))
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1
        importance = logits.sum(dim=(0))  # [3]
        loss_importance3 = self.cv2_squared(importance)
        lambda_2 = 	0.1
        loss = loss_importance3 + lambda_2 * loss_dynamic
        return top_p_mask, loss

    def forward(self, x, masks=None, is_training=None):
        # x [B, H, L, L]
        B, H, L, _ = x.shape  #[16,3,828]
        device = x.device
        dtype = torch.float32

        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            return mask_base, 0.0

        x = x.reshape(B * H, L, L)   #[16*4,828,828]
        gates, loss = self.noisy_top_k_gating(x, is_training)
        gates = gates.reshape(B, H, L, -1).float()

        mask = torch.einsum('bhli,lid->bhld', gates, masks) + mask_base
        return mask, loss


def mask_topk(x, alpha=0.5, largest=False):
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)  # 1 is topk
    return mask


class GraphLearner(nn.Module):
    def __init__(self, dim, n_vars, top_p=0.5, in_dim=12):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        # x: [B, L, D]

        adj = F.gelu(torch.einsum('bhid,bhjd->bhij', self.proj_1(x), self.proj_2(x)))
        adj = adj * mask_topk(adj, alpha)
        mask, loss = self.mask_moe(adj, masks, is_training)
        adj = adj * mask
        return adj, loss   # [B, L, L]

class Filter_Backbone(nn.Module):
    def __init__(self, dim, n_vars,n_heads=4,  top_p=0.5, dropout=0.5, in_dim=12):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        # x: [B, L, D]
        B, L, D = x.shape # x [16,207,64]
        adj, loss  = self.graph_learner(x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3), masks,
                                       alpha, is_training)  # [B, H, L, L]
        # keep = (adj != 0)                  # ← 这一行必须有
        # adj = adj.masked_fill(~keep, -1e9)
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)

        return adj, loss  # [B, L, D]
