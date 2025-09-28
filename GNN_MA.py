# -*- coding: UTF-8 -*-
'''
@Author  ：Iris
@Date    ：2025-09-28 18:40 
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 工具：masked softmax ----------
def masked_softmax(scores, mask, dim=-1):
    # mask: 1表示有效，0表示无效
    mask = mask.to(dtype=scores.dtype)
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(mask == 0, neg_inf)
    return F.softmax(scores, dim=dim)

# ==== 1. 节点-边联合卷积层（单图内） ====
class GraphNodeEdgeConvLayer(nn.Module):
    """
    输入:
      node_feat: [B, N, dn]
      adj:       [B, N, N] (0/1或权重)
      edge_feat: [B, N, N, de]
    输出:
      h_out:     [B, N, dh]
      e_out:     [B, N, N, dh]  (将边也编码到同一维度，便于后续跨边注意力)
    """
    def __init__(self, node_in, edge_in, hidden, dropout=0.5):
        super().__init__()
        self.lin_node = nn.Linear(node_in, hidden)
        self.lin_edge = nn.Linear(edge_in, hidden)
        # 用节点对更新边：phi([h_i, h_j, e_ij])
        self.lin_edge_update = nn.Linear(hidden*3, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, adj, edge_feat):
        B, N, _ = node_feat.shape
        # 邻居节点聚合
        agg_n = torch.bmm(adj, node_feat)             # [B, N, dn]
        hn = F.relu(self.lin_node(agg_n))             # [B, N, h]

        # 边特征聚合（按邻接求和）
        # edge_agg_i = sum_j A_ij * e_ij
        edge_agg = torch.einsum('bij,bijk->bik', adj, edge_feat)  # [B, N, de]
        he = F.relu(self.lin_edge(edge_agg))                       # [B, N, h]

        h_out = self.dropout(F.relu(hn + he))                      # [B, N, h]

        # —— 更新边嵌入（使用最新节点表征）——
        hi = h_out.unsqueeze(2).expand(B, N, N, h_out.size(-1))    # [B,N,N,h]
        hj = h_out.unsqueeze(1).expand(B, N, N, h_out.size(-1))    # [B,N,N,h]
        e_proj = F.relu(self.lin_edge(edge_feat))                  # [B,N,N,h]
        e_cat  = torch.cat([hi, hj, e_proj], dim=-1)               # [B,N,N,3h]
        e_out  = self.dropout(F.relu(self.lin_edge_update(e_cat))) # [B,N,N,h]

        # 仅保留存在的边（可选）
        e_out = e_out * adj.unsqueeze(-1)

        return h_out, e_out

# ==== 2. 跨图“点→点”注意力 ====
class CrossGraphNodeAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

    def forward(self, A, B, mask_B=None):
        # A: [B, NA, h], B: [B, NB, h]
        Q = self.q(A); K = self.k(B); V = self.v(B)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # [B, NA, NB]
        if mask_B is not None:
            # mask_B: [B, NB] -> [B,1,NB]
            attn = masked_softmax(scores, mask_B.unsqueeze(1).expand_as(scores), dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)  # [B, NA, h]
        return out

# ==== 3. 跨图“边→边”注意力 ====
class CrossGraphEdgeAttention(nn.Module):
    """
    将两图的边嵌入扁平化后做注意力：
      EA: [B, NA, NA, h] -> [B, NA*NA, h]
      EB: [B, NB, NB, h] -> [B, NB*NB, h]
    支持边mask（依据adj）
    """
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

    def forward(self, EA, EB, mask_EB=None):
        B, NA, _, H = EA.shape
        NB = EB.shape[1]
        EA_f = EA.reshape(B, NA*NA, H)   # [B, NA^2, h]
        EB_f = EB.reshape(B, NB*NB, H)   # [B, NB^2, h]

        Q = self.q(EA_f)
        K = self.k(EB_f)
        V = self.v(EB_f)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # [B, NA^2, NB^2]
        if mask_EB is not None:
            # mask_EB: [B, NB, NB] -> [B, NB^2]
            mask_EB_f = mask_EB.reshape(B, NB*NB)
            scores = scores.masked_fill(mask_EB_f.unsqueeze(1) == 0, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        EA_agg_f = torch.bmm(attn, V)                      # [B, NA^2, h]
        EA_agg   = EA_agg_f.reshape(B, NA, NA, H)          # [B, NA, NA, h]
        return EA_agg

# ==== 4. 边→点 聚合（把边级信息还给节点） ====
class EdgeToNode(nn.Module):
    def __init__(self, hidden, undirected=True):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.undirected = undirected

    def forward(self, E, adj):
        # E: [B, N, N, h]
        # 对于每个节点 i，聚合 incident edges：sum_j E[i,j] (+ E[j,i] if undirected)
        inc_out = torch.einsum('bijk,bij->bik', E, adj)  # sum_j E[i,j]*A_ij
        if self.undirected:
            inc_out = inc_out + torch.einsum('bjik,bij->bik', E, adj)  # sum_j E[j,i]*A_ij
        return F.relu(self.lin(inc_out))  # [B, N, h]

# ==== 5. GMN 主干（点+边双跨图注意力） ====
class GraphMatchingNetwork(nn.Module):
    """
    forward参数：
      x1:  [B, N1, dn], adj1: [B, N1, N1], e1: [B, N1, N1, de], mask1: [B, N1] (可选)
      x2:  [B, N2, dn], adj2: [B, N2, N2], e2: [B, N2, N2, de], mask2: [B, N2] (可选)
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphNodeEdgeConvLayer(node_dim if i==0 else hidden_dim,
                                   edge_dim if i==0 else hidden_dim,
                                   hidden_dim, dropout)
            for i in range(num_layers)
        ])
        self.cross_node = CrossGraphNodeAttention(hidden_dim)
        self.cross_edge = CrossGraphEdgeAttention(hidden_dim)
        self.edge2node  = EdgeToNode(hidden_dim, undirected=True)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, adj1, e1, x2, adj2, e2, mask1=None, mask2=None):
        h1, h2 = x1, x2
        E1, E2 = e1, e2  # 显式边特征

        for layer in self.layers:
            # 单图内更新（节点+边）
            h1, E1 = layer(h1, adj1, E1)
            h2, E2 = layer(h2, adj2, E2)

            # —— 跨图“点→点”注意力（带mask）——
            cross_h1 = self.cross_node(h1, h2, mask_B=mask2)  # [B, N1, h]
            cross_h2 = self.cross_node(h2, h1, mask_B=mask1)  # [B, N2, h]

            # —— 跨图“边→边”注意力（依据adj做边mask）——
            cross_E1 = self.cross_edge(E1, E2, mask_EB=adj2)  # [B, N1, N1, h]
            cross_E2 = self.cross_edge(E2, E1, mask_EB=adj1)  # [B, N2, N2, h]

            # —— 边→点 聚合，把跨边信息反馈到节点 ——
            edge_msg_1 = self.edge2node(cross_E1, adj1)       # [B, N1, h]
            edge_msg_2 = self.edge2node(cross_E2, adj2)       # [B, N2, h]

            # 残差融合
            h1 = h1 + cross_h1 + edge_msg_1
            h2 = h2 + cross_h2 + edge_msg_2

        # -------- 池化（支持mask） --------
        if mask1 is not None:
            g1 = (h1 * mask1.unsqueeze(-1)).sum(dim=1) / (mask1.sum(dim=1, keepdim=True) + 1e-9)
        else:
            g1 = h1.mean(dim=1)
        if mask2 is not None:
            g2 = (h2 * mask2.unsqueeze(-1)).sum(dim=1) / (mask2.sum(dim=1, keepdim=True) + 1e-9)
        else:
            g2 = h2.mean(dim=1)

        out = torch.cat([g1, g2], dim=-1)
        score = self.readout(out)  # [B, 1]
        return score
