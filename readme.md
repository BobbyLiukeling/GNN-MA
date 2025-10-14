# GNN-MA: Graph Matching Network for 3D Molecular Similarity & Virtual Screening

GNN-MA is a graph neural network framework for 3D ligand similarity alignment and virtual screening. It supports cross - graph attention, node - edge collaborative convolution, and evaluation on multiple datasets (DUD - E, LIT - PCBA/PCBA). It also provides complete data pre - processing, training, and evaluation scripts. 
For the complete code, please refer to: https://github.com/BobbyLiukeling/GNN - MA

## 1. Environmental Requirements

OS：Linux / macOS / Windows(Linux is recommended) are all acceptable.

Python：3.10

Dependency Installation (Conda Recommended)

1. numpy                             1.23.5
2. pandas                            2.2.3
3. rdkit                             2025.3.4
4. torch                             2.5.1
5. matplotlib                        3.7.2

## 2. Data Download and Pre - processing
This model adopts the virtual screening benchmark datasets DUD - E and LIT - PCBA, and the data will be processed into NPZ format before training.
1. Download from the official website of the virtual screening benchmark dataset DUD - E (https://dude.docking.org/) and the website of LIT - PCBA (http://drugdesign.unistra.fr/LIT - PCBA) respectively, and store them in the DUD - E and LIT - PCBA folders. Due to storage space limitations, there is only one protein target in both DUD - E and LIT - PCBA. You can download the complete data and place it in the specified folder for training.
2. Process the DUD - E data, split the merged decoy molecules, use split_DUDE.py for processing, and store the processed data in the DUD - E - dealed folder.
3. Encode the molecules in the DUD - E and LIT - PCBA datasets, encode different types of molecular data into npz format, use encoding.py and encoding - LIT.py for processing respectively, and store them in the encode - DUE - E and encode - LIT folders after processing.
4. All data - related processing codes and files are stored in the data folder.

## 3. GNN-MA Model Construction
GNN-MA is a graph neural network model designed for three-dimensional molecular similarity alignment. The model is based on a graph convolutional neural network, augmented with a cross-graph attention mechanism to capture inter-molecular dependencies, enabling the molecular of atomic, bond, and pharmacophore features for molecular alignment and similarity evaluation.
GNN-MA supports molecular comparison and alignment at multiple dimension, making it broadly applicable to structure-based drug design tasks.
In this study, molecular structures are represented as undirected weighted graphs , where the graph node set H corresponds to atoms and the edge set  corresponds to chemical bonds. The connectivity between atoms is encoded by an adjacency matrix , where  denotes the number of atoms in a molecule. Training samples are provided as molecular pairs, meaning that the model simultaneously receives two molecular graphs,  and , as input. Each molecular is described by three core components: an atom(node) feature matrix, a bond(edge) feature tensor, and an adjacency matrix. We adopt one-hot encoding to uniformly represent all feature. Based on extensive prior studies and our comparative experiments, we selected the following descriptors to represent atomic and bond features.

The specific content is implemented in GNN-MA.py.

<img src="Figure 1.png" alt="Alternative Text" width="1168" height="458">



```python
# Tools
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Tool：masked softmax ----------
def masked_softmax(scores, mask, dim=-1):
    # mask: 1indicates valid, 0 indicates invalid
    mask = mask.to(dtype=scores.dtype)
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(mask == 0, neg_inf)
    return F.softmax(scores, dim=dim)
```

**3.2.1 Intra-Graph Message Passing**

Within a single molecule, atom and bond are updated simultaneously through a cooperative convolution process, as illustrated in Figure 1(b).

(1) Neighborhood aggregation for atoms and bonds follows eqs 1 and 2.
$$
l(h)=\varphi(AH\,W_h)                                    \tag{1}
$$

$$
l(e)=\varphi(AE\,W_e)                                    \tag{2}
$$

Where, $W_h$ and $W_e$ are learnable weight parameters for atom and bond, respectively. $\ell^{(h)}$ and $\ell^{(e)}$ denote the aggregated atom and bond information, and $\varphi$ represents the activation function.

(2) Fusion of node and edge information

Node fusion is defined in eq 3				
$$
\widetilde{H}=\mathrm{Dropout}\,\varphi\big(\ell^{(h)}+\ell^{(e)}\big) \tag{3}
$$

Edge fusion is defined in eq 4
$$
\widetilde{E}=f_{\text{mask}}\big(A_{ij},\ f_{\text{edge}}(\widetilde{H}_i,\widetilde{H}_j,e_{ij})\big)\big) \tag{4}
$$
Here, $\widetilde{H}_i$ and $\widetilde{H}_j$ denote the information of the i-th and j-th atoms, respectively. The function $f_{\text{edge}}$ fuses the bond feature $e_{ij}$ with the updated features of atom and . The function $f_{\text{mask}}$ applies a linear transformation to the bond representation using the adjacency matrix.




```python
# ==== 1. Node-Edge Joint Convolution Layer (Within a Single Graph) ====
class GraphNodeEdgeConvLayer(nn.Module):
    """
    Inuput:
      node_feat: [B, N, dn]
      adj:       [B, N, N] (0/1 or weights)
      edge_feat: [B, N, N, de]
    Output:
      h_out:     [B, N, dh]
      e_out:     [B, N, N, dh]  (Encode edges to the same dimension to facilitate subsequent cross-edge attention)
    """
    def __init__(self, node_in, edge_in, hidden, dropout=0.5):
        super().__init__()
        self.lin_node = nn.Linear(node_in, hidden)
        self.lin_edge = nn.Linear(edge_in, hidden)
        # Update edges with node pairs：phi([h_i, h_j, e_ij])
        self.lin_edge_update = nn.Linear(hidden*3, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, adj, edge_feat):
        B, N, _ = node_feat.shape
        # Neighbor node aggregation
        agg_n = torch.bmm(adj, node_feat)             # [B, N, dn]
        hn = F.relu(self.lin_node(agg_n))             # [B, N, h]

        # Edge feature aggregation (sum by adjacency)
        # edge_agg_i = sum_j A_ij * e_ij
        edge_agg = torch.einsum('bij,bijk->bik', adj, edge_feat)  # [B, N, de]
        he = F.relu(self.lin_edge(edge_agg))                       # [B, N, h]

        h_out = self.dropout(F.relu(hn + he))                      # [B, N, h]

        # —— Update edge embeddings (using the latest node representations)——
        hi = h_out.unsqueeze(2).expand(B, N, N, h_out.size(-1))    # [B,N,N,h]
        hj = h_out.unsqueeze(1).expand(B, N, N, h_out.size(-1))    # [B,N,N,h]
        e_proj = F.relu(self.lin_edge(edge_feat))                  # [B,N,N,h]
        e_cat  = torch.cat([hi, hj, e_proj], dim=-1)               # [B,N,N,3h]
        e_out  = self.dropout(F.relu(self.lin_edge_update(e_cat))) # [B,N,N,h]

        # Only retain the existing edges (optional)
        e_out = e_out * adj.unsqueeze(-1)

        return h_out, e_out

```

### 3.2.2 Cross-graph attention
Given two molecular graphs $G_1$ and $G_2$, the representation of $G_1$ is updated by integrating information from $G_2$, thereby enabling $G_1$ to capture cross-graph dependencies, as illustrated in Figure 1(c).

1.  Computation of atoms attention scores is defined in eq 5.
$$
Q_{G_1}^{(h)}=\widetilde{H}_{G_1}W_q^{(h)},\ \ K_{G_2}^{(h)}=\widetilde{H}_{G_2}W_k^{(h)},\ \ V_{G_2}^{(h)}=\widetilde{H}_{G_2}W_v^{(h)} \tag{5}
$$


Where $W_q^{(h)}$, $W_k^{(h)}$, $W_v^{(h)}$ and are learnable weight matrices, while $\widetilde{H}_{G_1}$ and $\widetilde{H}_{G_2}$ denote the atom features of $G_1$ and $G_2$ after intra molecular graph convolution.

The atoms information in $G_2$ is then aggregated into the atoms of $G_1$ according to eq 6:
$$
\overleftrightarrow{H}_{G_1 \leftarrow G_2}
= \operatorname{softmax}\!\left(
  \frac{\,Q_{G_1}^{(h)} \big(K_{G_2}^{(h)}\big)^{\top}}{\sqrt{d_h}}
\right) V_{G_2}^{(h)} \tag{6}
$$


```python
# ==== 2. Cross-Graph "Node-to-Node" Attention ====
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

```


2. Computation of bonds attention scores is defined in eq 7.
$$
Q^{(e)}_{G_1}=\widetilde{E}_{G_1}W^{(e)}_{q},\quad
K^{(e)}_{G_2}=\widetilde{E}_{G_2}W^{(e)}_{k},\quad
V^{(e)}_{G_2}=\widetilde{E}_{G_2}W^{(e)}_{v} \tag{7}
$$

Here, $W^{(e)}_{q}$ $W^{(e)}_{k}$ and $W^{(e)}_{v}$ are learnable weight matrices, while $\widetilde{E}_{G_1}$ and $\widetilde{E}_{G_2}$ denote the bond features of $G_1$ and $G_2$ after intra molecular graph convolution.

The bonds information from $G_2$ is then aggregated into the bonds of $G_1$ according to eq 8:
$$
\overleftrightarrow{E}_{G_1 \leftarrow G_2}
= \operatorname{softmax}\!\left(
  \frac{\,Q^{(e)}_{G_1}\big(K^{(e)}_{G_2}\big)^{\top}}{\sqrt{d_h}}
\right)\, V^{(e)}_{G_2} \tag{8}
$$


```python

# ==== 3. Cross-Graph "Edge-to-Edge" Attention ====
class CrossGraphEdgeAttention(nn.Module):
    """
    After flattening the edge embeddings of the two graphs, perform attention computation:
      EA: [B, NA, NA, h] -> [B, NA*NA, h]
      EB: [B, NB, NB, h] -> [B, NB*NB, h]
    Supports edge masking（based on adjacency matrix）
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
```

### 3.2.3 bond to atom Aggregation

Within each molecule, the updated bond information through cross molecular graph attention is aggregated to the corresponding atoms.

First, for a given atom $i$, the information from all bonds connected to it is aggregated as follow eqs 9 and 10:
$$
				m_i=\sum_{j=1}^{N} A_{ij}(\overleftrightarrow{E}_{ij} + \overleftrightarrow{E}_{ij}) \tag{9}
$$

$$
				M=\begin{bmatrix} m_1 \\ \vdots \\ m_N \end{bmatrix} \in \mathbb{R}^{N\times h} \tag{10}
$$

Here,$\overleftrightarrow{E}_{ij}$ represents the bond between atoms $i$ and $j$ , which are refined through graph convolution and cross graph attention.

Finally, aggregated all atom information is defined in eq 11:
$$
				H^{(e\to n)}=\varphi(MW) \tag{11}
$$


where $w$ is a trainable weight matrix.



```python

# ==== 4. Edge-to-Node Aggregation (Return edge-level information to nodes) ====
class EdgeToNode(nn.Module):
    def __init__(self, hidden, undirected=True):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.undirected = undirected

    def forward(self, E, adj):
        # E: [B, N, N, h]
        # For each node i, aggregate  incident edges：sum_j E[i,j] (+ E[j,i] if undirected)
        inc_out = torch.einsum('bijk,bij->bik', E, adj)  # sum_j E[i,j]*A_ij
        if self.undirected:
            inc_out = inc_out + torch.einsum('bjik,bij->bik', E, adj)  # sum_j E[j,i]*A_ij
        return F.relu(self.lin(inc_out))  # [B, N, h]

```

### **3.2.4 Data fusion and get similarity scoring**

After intra graph convolution, cross graph attention, and bond to atom message passing, the updated molecular signal are integrated through residual aggregation is defined in eq 12, and as illustrated in Figure 1(d):

$$
				G=\widetilde{H}+\overleftrightarrow{H}+H^{(e\to n)} \tag{12}
$$

For similarity evaluation, the two molecular graphs $G_1$ and $G_2$ are concatenated to form a joint representation is defined in eq 13:
$$
				z=[G1\,||\,G2] \tag{13}
$$
where "$||$" denotes the feature concatenation operator. The similarity score is then computed is defined in eq 14:
$$
				s=w_2^{\top}\, \mathrm{Dropout}\!\big( \varphi(W_1 z + b_1) \big) + b_2 \tag{14}
$$


Here, $W_2,W_1, b_1$ and $b_2$ are learnable parameters, and represents the final similarity score.



```python

# ==== 5. GMN Backbone (Node + Edge Cross-Graph Attention) ====
class GraphMatchingNetwork(nn.Module):
    """
    forward parameters：
      x1:  [B, N1, dn], adj1: [B, N1, N1], e1: [B, N1, N1, de], mask1: [B, N1] (optional)
      x2:  [B, N2, dn], adj2: [B, N2, N2], e2: [B, N2, N2, de], mask2: [B, N2] (optional)
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
        E1, E2 = e1, e2  # Explicit edge features

        for layer in self.layers:
            # Intra-graph update (node + edge)
            h1, E1 = layer(h1, adj1, E1)
            h2, E2 = layer(h2, adj2, E2)

            # —— Cross-graph "node-to-node" attention (with mask)——
            cross_h1 = self.cross_node(h1, h2, mask_B=mask2)  # [B, N1, h]
            cross_h2 = self.cross_node(h2, h1, mask_B=mask1)  # [B, N2, h]

            # —— Cross-graph "edge-to-edge" attention (perform edge masking based on adjacency matrix)——
            cross_E1 = self.cross_edge(E1, E2, mask_EB=adj2)  # [B, N1, N1, h]
            cross_E2 = self.cross_edge(E2, E1, mask_EB=adj1)  # [B, N2, N2, h]

            # —— Edge-to-node aggregation: feed cross-edge information back to nodes ——
            edge_msg_1 = self.edge2node(cross_E1, adj1)       # [B, N1, h]
            edge_msg_2 = self.edge2node(cross_E2, adj2)       # [B, N2, h]

            # Residual fusion
            h1 = h1 + cross_h1 + edge_msg_1
            h2 = h2 + cross_h2 + edge_msg_2

        # -------- Pooling (support masking) --------
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

```

## 训练
The model was trained in a supervised learning framework, Each training sample was formed by combining a ligand with either an active or a decoy compound. Binary cross-entropy loss was employed as the loss function , and model parameters were optimized using the Adam algorithm. The initial learning rate was set to ,batch size is 32, and training proceeded for 20 epochs.

Since the DUD - E and LIT - PCAB datasets are organized in different ways, the codes used during training also have some differences. Among them, train.py is the code for training the model on the DUD - E dataset, and train - LIT.py is the code for training the model on the LIT - PCBA dataset. A GPU can be used during training, and a CPU can be used as a substitute on devices without a GPU.
