import torch
import torch.nn as nn
import numpy as np
from model.Embed import PositionalEmbedding
from model.Filter_layers import Filter_Backbone
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)
    def forward(self, x):
        # x: [B, C*T]
        B,T,N,C = x.shape
        x_n = x[..., 0]
        x_m = x[..., 0]
        x_m = x_m.reshape(-1, T * N)  # [B, N*T]
        x_n = x_n.permute(0, 2, 1).reshape(-1, N * T)  # [B, N*T]
        x_n = x_n.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [B,N*L,P] 16,207*4,4

        x_m = x_m.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [B*c,N*T/P] 16*3,207*4,3
        # x: [B, N, P]
        x_n = self.patch_proj(x_n)  # [B, N, D]
        if self.pos:
            x_n += self.pe(x_n)
        # x_m = self.patch_proj(x_m)  # [B, N, D]
        # if self.pos:
        #     x_m += self.pe(x_m)
        return x_n,x_m

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None
        self.proj = nn.Linear(output_dim, output_dim)
        self.proj1 = nn.Linear(input_dim, output_dim)
        self.proj2 = nn.Linear(output_dim, input_dim)
        self.proj3 = nn.Linear(32, 67)
        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        x_g = []
        if support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        elif support.dim() == 4:
         # support shape is [B, H, L, L]  此处L为N
            H = support.shape[1]
            B, L, D = x.shape
            x = self.proj1(x)
            x = self.proj(x).view(B, L, H, -1) # [B, L, H, D_]会对x
            B,L,H,W = x.shape
            identity = torch.eye(support.shape[2], device=support.device).repeat(B, H, 1, 1)
            graph_list = [identity, support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            x_permuted = x.permute(0, 2, 1, 3)  # Shape becomes [B, H, L, D_]
            B, L, H, D_ = x.shape
            for graph in graph_list:
                convolved_x= torch.einsum('bhij,bhjd->bhid', graph, x_permuted)
                convolved_x_3d = convolved_x.permute(0, 2, 1, 3).reshape(B, L, H * D_)
                convolved_x_3d = self.proj2(convolved_x_3d)
                x_g.append(convolved_x_3d)
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv


class GCRU(nn.Module):
    def __init__(
        self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = GCN(
            input_dim + self.hidden_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis
        )
        self.update = GCN(
            input_dim + self.hidden_dim, output_dim, cheb_k, embed_dim, meta_axis
        )
    def forward(self, x, state, support, embeddings):
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        h = r * state + (1 - r) * hc
        return h
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)


class Encoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="S",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [GCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                GCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x, support, embeddings):
        # x: (B, T, N, C)
        batch_size = x.shape[0]
        in_steps = x.shape[1]

        current_input = x
        output_hidden = []
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size).to(x.device)
            inner_states = []
            for t in range(in_steps):
                state = cell(current_input[:, t, :, :], state, support, embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)
        return current_input, output_hidden


class Decoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="ST",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [GCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                GCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, xt, init_state, support, embeddings):
        current_input = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_input, init_state[i], support, embeddings)
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden

class AttentionPooling(nn.Module):
    def __init__(self, num_nodes, patch_per_node=4):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(patch_per_node, 1)
        )
        self.num_nodes = num_nodes

    def forward(self, adj):
        B, H, L, _ = adj.shape
        N = self.num_nodes
        P = L // N

        attn_w = torch.softmax(self.weight, dim=0)

        adj_rows = adj.view(B, H, L, N, P)
        adj_cols = torch.matmul(
            adj_rows, attn_w
        ).squeeze(-1)

        adj_final = (
            adj_cols.view(B, H, N, P, N)
            .permute(0, 1, 2, 4, 3)
        )
        return torch.matmul(
            adj_final, attn_w
        ).squeeze(-1)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        # self.qkv_norm = nn.LayerNorm(dim)
    def forward(self, q, k, v, mode=None):
        B, T, N, D = q.shape
        q = self.q_norm(self.q_proj(q))
        k = self.k_norm(self.k_proj(k))
        v = self.v_proj(v)
        if mode == "time":
            # 每个节点独立做时间注意力
            q = q.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # B,H,N,D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, N, T, D).permute(0, 2, 1, 3)
        else:  # "space"
            q = q.view(B * T, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(B * T, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(B * T, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, T, N, D)
        return self.out_proj(out)

class DyEGFCA_Net(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim=3,
        output_dim=1,
        out_steps=12,
        hidden_dim=64,
        num_layers=1,
        cheb_k=2,
        ycov_dim=2,
        tod_embedding_dim=8,
        dow_embedding_dim=8,
        node_embedding_dim=16,
        st_embedding_dim=16,
        tf_decay_steps=6000,
        use_teacher_forcing=True,
        use_pos_embed=True,
        n_heads=4,
        stride=3,
        patch_len=3,
        top_p=0.5,
        alpha=0.1,
        dropout_adj = 0.3,
        input_steps = 12,
    ):
        super().__init__()
        self.input_steps = input_steps
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.out_steps = out_steps
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.node_embedding_dim = node_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing
        self.patch_len = patch_len
        self.dropout_adj = dropout_adj

        self.encoder_s = Encoder(
            num_nodes,
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            node_embedding_dim,
            meta_axis="S",
        )
        self.encoder_t = Encoder(
            num_nodes,
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            tod_embedding_dim + dow_embedding_dim,
            meta_axis="T",
        )

        self.decoder = Decoder(
            num_nodes,
            output_dim + ycov_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            st_embedding_dim,
        )
        self.top_p = top_p
        self.alpha = alpha
        self.patch_embed = PatchEmbed(hidden_dim, self.patch_len, stride, pos=use_pos_embed)
        self.backbone = Filter_Backbone(hidden_dim, num_nodes,  n_heads, self.top_p,self.dropout_adj,
                        self.input_steps * num_nodes // self.patch_len, )
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.t2s_attn = CrossAttention(hidden_dim)
        self.s2t_attn = CrossAttention(hidden_dim)
        self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.node_embedding = nn.init.xavier_normal_(
            nn.Parameter(torch.empty(self.num_nodes, self.node_embedding_dim))
        )
        self.st_proj = nn.Linear(self.hidden_dim, self.st_embedding_dim)
        self.start_adj= nn.Conv1d(in_channels=self.num_nodes * self.input_steps//self.patch_len,
                                  out_channels=self.num_nodes,kernel_size=1,bias=True)
        self.dropout = nn.Dropout(0.3)

    def compute_sampling_threshold(self, batches_seen):
        return self.tf_decay_steps / (
            self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self, x, y_cov,masks,labels=None, batches_seen=None,is_training = False, return_case_study=False):

        x_n, x_m = self.patch_embed(x)  # [B, N, D]  N = [N*T / P]  D是由Patch的原始维度变换为隐藏维度  x_n[16,828,64]   x_m[16,828,3,3]
        adj, moe_loss= self.backbone(x_n, masks, self.alpha, is_training)  # [16,4,828,828]
        B,H,p,_ = adj.shape
        adj = adj.view(B*H,p,p)
        adj = self.start_adj(adj)
        adj = adj.permute(0,2,1)
        adj = self.start_adj(adj)
        adj = adj.permute(0,2,1)
        BH,nl,_ = adj.shape
        adj = adj.view(B,H,nl,nl)
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)
        tod = x[:, -1, 0, 1]
        dow = x[:, -1, 0, 2]
        tod_emb = self.tod_embedding(
            (tod * 288).long()
        )  # (batch_size, tod_embedding_dim)
        dow_emb = self.dow_embedding(dow.long())  # (batch_size, dow_embedding_dim)
        time_embedding = torch.cat([tod_emb, dow_emb], dim=-1)  # (B, tod+dow)
        h_s, _ = self.encoder_s(x,adj, self.node_embedding)  # B, T, N, hidden
        h_t, _ = self.encoder_t(x, adj, time_embedding)  # B, T, N, hidden
        t2s_feat = self.t2s_attn(h_t, h_s, h_s, mode="space")  # (B, T, N, D)
        s2t_feat = self.s2t_attn(h_s, h_t, h_t, mode="time")  # (B, T, N, D)
        h_s_enh = h_s + t2s_feat
        h_t_enh = h_t + s2t_feat
        h_last = (h_s_enh + h_t_enh)[:, -1, :, :]  # [16,N,64]
        st_embedding = self.st_proj(h_last)  # B, N, st_emb_dim
        support = torch.softmax(
            torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding, st_embedding)),
            dim=-1,
        )
        ht_list = [h_last] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.out_steps):
            h_de, ht_list = self.decoder(
                torch.cat([go, y_cov[:, t, ...]], dim=-1),
                ht_list,
                support,
                st_embedding,
            )
            go = self.out_proj(h_de)
            out.append(go)
            if self.training and self.use_teacher_forcing:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
        return output, moe_loss,

if __name__ == "__main__":
    from torchinfo import summary

    model = DyEGFCA_Net(num_nodes=207).gpu()
    # 创建虚拟输入张量
    x_dummy = torch.zeros(16, 12, 207, 3)  # [B, T, N, C]
    y_cov_dummy = torch.zeros(16, 12, 207, 2)  # [B, T, N, ycov_dim]

    # 创建其他参数的虚拟值
    labels_dummy = None
    batches_seen_dummy = 0
    masks_dummy = None
    is_training_dummy = False

    # 调用 summary
    summary(
        model,
        input_data=(
            x_dummy,
            y_cov_dummy,
            masks_dummy,
            labels_dummy,
            batches_seen_dummy,

            is_training_dummy
        ),
        device="cpu"
    )
