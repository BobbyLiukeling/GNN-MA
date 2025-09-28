# -*- coding: UTF-8 -*-
'''
@Author  ：Iris
@Date    ：2025-09-28 18:47 

说明：
- 适配“每个靶标有多个 ligands/actives/decoys”的新数据集；
- 对每个靶标内：每个 ligand × 每个 active 生成正例；每个 ligand × 每个 decoy 生成负例；
- 兼容目录：
    {ENCODE_ROOT}/{target}/{MODE}/ligand/*.npz   或 ligands/*.npz
    {ENCODE_ROOT}/{target}/{MODE}/active/*.npz
    {ENCODE_ROOT}/{target}/{MODE}/decoy/*.npz
  也兼容历史：{MODE}/crystal_ligand*.npz（单或多）
'''
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time
import csv
import warnings
warnings.filterwarnings("ignore")

from GNN_MA import GraphMatchingNetwork


# ========= 0. 小工具 =========
def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ========= 1. 数据集与数据加载 =========
class PairGraphDataset(Dataset):
    """
    pair 结构为 dict：
      {
        'target':  靶标名,
        'file':    另一分子 .npz 路径 (active/decoy),
        'label':   0/1,
        'lig_node','lig_adj','lig_edge': 当前 ligand 的图 (左图),
        'lig_file': 当前 ligand 的文件路径,
        'pair_type': 'pos' 或 'neg'
      }
    """
    def __init__(self, pair_list):
        self.pair_list = pair_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        p = self.pair_list[idx]
        lig_node = p['lig_node']; lig_adj = p['lig_adj']; lig_edge = p['lig_edge']
        node1 = lig_node.astype(np.float32)
        adj1  = lig_adj.astype(np.float32)
        e1    = lig_edge.astype(np.float32) if lig_edge is not None else adj1[..., None]

        dat = np.load(p['file'])
        node2 = dat['node_feat'].astype(np.float32)
        adj2  = dat['adj'].astype(np.float32)
        e2    = dat['edge_feat'].astype(np.float32) if 'edge_feat' in dat else adj2[..., None]

        label = np.array([p['label']], dtype=np.float32)
        meta = {
            'target': p['target'],
            'file': p['file'],
            'label': int(p['label']),
            'lig_file': p.get('lig_file', ''),
            'pair_type': p.get('pair_type', 'unk')
        }
        return (node1, adj1, e1, node2, adj2, e2, label, meta)


def pad_collate(batch):
    # 计算 pad 尺寸
    max_n1 = max(x[0].shape[0] for x in batch)
    max_n2 = max(x[3].shape[0] for x in batch)

    xs1, adjs1, es1, xs2, adjs2, es2, masks1, masks2, labels, metas = [], [], [], [], [], [], [], [], [], []
    for x1, adj1, e1, x2, adj2, e2, label, meta in batch:
        n1, n2 = x1.shape[0], x2.shape[0]
        # 节点特征
        pad_x1 = np.pad(x1, ((0, max_n1-n1), (0, 0)), 'constant')
        pad_x2 = np.pad(x2, ((0, max_n2-n2), (0, 0)), 'constant')
        # 邻接
        pad_adj1 = np.pad(adj1, ((0, max_n1-n1), (0, max_n1-n1)), 'constant')
        pad_adj2 = np.pad(adj2, ((0, max_n2-n2), (0, max_n2-n2)), 'constant')
        # 边特征 [N,N,de]
        pad_e1 = np.pad(e1, ((0, max_n1-n1), (0, max_n1-n1), (0, 0)), 'constant')
        pad_e2 = np.pad(e2, ((0, max_n2-n2), (0, max_n2-n2), (0, 0)), 'constant')

        mask1 = np.zeros(max_n1, dtype=np.float32); mask1[:n1] = 1
        mask2 = np.zeros(max_n2, dtype=np.float32); mask2[:n2] = 1

        xs1.append(pad_x1); adjs1.append(pad_adj1); es1.append(pad_e1)
        xs2.append(pad_x2); adjs2.append(pad_adj2); es2.append(pad_e2)
        masks1.append(mask1); masks2.append(mask2)
        labels.append(label); metas.append(meta)

    return (
        torch.tensor(xs1,  dtype=torch.float32),  # x1 [B,N1,dn]
        torch.tensor(adjs1,dtype=torch.float32),  # adj1 [B,N1,N1]
        torch.tensor(es1,  dtype=torch.float32),  # e1 [B,N1,N1,de]
        torch.tensor(xs2,  dtype=torch.float32),
        torch.tensor(adjs2,dtype=torch.float32),
        torch.tensor(es2,  dtype=torch.float32),
        torch.tensor(masks1,dtype=torch.float32), # mask1 [B,N1]
        torch.tensor(masks2,dtype=torch.float32),
        torch.tensor(labels,dtype=torch.float32), # [B,1]
        metas
    )


# ========= 2. Pair 采集与划分（多 ligand 适配） =========
def _list_npz_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')])

def _collect_ligand_files(folder):
    """
    优先在 {folder}/ligand 或 ligands 下找 .npz；
    若不存在，则回退到 {folder} 下的 crystal_ligand*.npz（兼容旧版）。
    """
    lig_dirs = [os.path.join(folder, 'ligand'), os.path.join(folder, 'ligands')]
    lig_files = []
    for d in lig_dirs:
        lig_files.extend(_list_npz_files(d))
    if lig_files:
        return lig_files

    # 兼容历史命名：直接在 folder 里放 crystal_ligand*.npz（可能是1个或多个）
    legacy = []
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith('ligand.npz'):
                legacy.append(os.path.join(folder, f))
    return sorted(legacy)

def get_graph_pairs_from_target_multi(
    target_dir, mode, target_name,
    max_pos_per_ligand=None,         # 可选：每个 ligand 限制最多正例数量
    max_neg_per_ligand=None          # 可选：每个 ligand 限制最多负例数量
):
    """
    在单一靶标下做“每个 ligand × 每个 active/decoy”的全配对（或限流配对）
    """
    folder = os.path.join(target_dir, mode)

    lig_files = _collect_ligand_files(folder)
    act_files = _list_npz_files(os.path.join(folder, 'active'))
    dec_files = _list_npz_files(os.path.join(folder, 'decoy'))

    if len(lig_files) == 0:
        # 既没有 ligand 目录，也没有 legacy 文件，跳过该靶标
        return []

    all_pairs = []
    # 逐个 ligand 读取一次并复用其 numpy 数组（避免重复加载）
    for lf in lig_files:
        lig_npz = np.load(lf)
        lig_node = lig_npz['node_feat']
        lig_adj  = lig_npz['adj']
        lig_edge = lig_npz['edge_feat'] if 'edge_feat' in lig_npz else None

        # 选取 active/decoy 列表（可选限流）
        pos_list = act_files if max_pos_per_ligand is None else act_files[:max_pos_per_ligand]
        neg_list = dec_files if max_neg_per_ligand is None else dec_files[:max_neg_per_ligand]

        # 正例
        for f in pos_list:
            all_pairs.append({
                'target': target_name, 'file': f, 'label': 1,
                'lig_node': lig_node, 'lig_adj': lig_adj, 'lig_edge': lig_edge,
                'lig_file': lf, 'pair_type': 'pos'
            })
        # 负例
        for f in neg_list:
            all_pairs.append({
                'target': target_name, 'file': f, 'label': 0,
                'lig_node': lig_node, 'lig_adj': lig_adj, 'lig_edge': lig_edge,
                'lig_file': lf, 'pair_type': 'neg'
            })

    random.shuffle(all_pairs)
    return all_pairs


def build_dataset(
    encode_root, mode,
    split_ratio=(0.8,0.1,0.1),
    max_pairs_per_target=None,       # 可选：对“单靶标生成的全部 pair”再做上限截断
    max_pos_per_ligand=None,         # 可选：每 ligand 正例上限
    max_neg_per_ligand=None          # 可选：每 ligand 负例上限
):
    """
    汇总所有靶标的 pair，随机划分为 train/val/test
    """
    all_targets = [d for d in os.listdir(encode_root) if os.path.isdir(os.path.join(encode_root, d))]
    all_pairs = []
    for t in all_targets:
        target_dir = os.path.join(encode_root, t)
        if not os.path.exists(os.path.join(target_dir, mode)):
            continue
        pairs = get_graph_pairs_from_target_multi(
            target_dir, mode, target_name=t,
            max_pos_per_ligand=max_pos_per_ligand,
            max_neg_per_ligand=max_neg_per_ligand
        )
        if max_pairs_per_target is not None and len(pairs) > max_pairs_per_target:
            random.shuffle(pairs)
            pairs = pairs[:max_pairs_per_target]
        all_pairs.extend(pairs)

    print(f"总pair数: {len(all_pairs)}")
    random.shuffle(all_pairs)
    n_total = len(all_pairs)
    n_train = int(n_total * split_ratio[0])
    n_val   = int(n_total * split_ratio[1])
    train = all_pairs[:n_train]
    val   = all_pairs[n_train:n_train+n_val]
    test  = all_pairs[n_train+n_val:]
    return train, val, test


# ========= 3. 训练循环 =========
def train_model(model, train_dl, val_dl, test_dl, device, out_model_prefix, epochs=20, lr=1e-3):
    save_dir = "save_data/edge_no_sidechain_cross_double_multiLig"
    os.makedirs(save_dir, exist_ok=True)
    prefix = os.path.join(save_dir, os.path.basename(out_model_prefix))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc = 0.0
    epoch_aucs, epoch_times = [], []

    # ---------- 训练 ----------
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        losses = []
        temp = -1
        for batch in train_dl:
            x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, _ = batch
            x1=x1.to(device); adj1=adj1.to(device); e1=e1.to(device)
            x2=x2.to(device); adj2=adj2.to(device); e2=e2.to(device)
            mask1=mask1.to(device); mask2=mask2.to(device); label=label.to(device)

            pred = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

            # 简单进度输出（每 ~1000ms 一次）
            t1 = int((time.time() - t0) * 1000)
            if t1 // 1000 > temp:
                temp = t1 // 1000
                if temp > 0 and temp % 100 == 0:
                    print(f"  {temp}s elapsed...")

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # ---------- 验证 ----------
        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for batch in val_dl:
                x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, _ = batch
                x1=x1.to(device); adj1=adj1.to(device); e1=e1.to(device)
                x2=x2.to(device); adj2=adj2.to(device); e2=e2.to(device)
                mask1=mask1.to(device); mask2=mask2.to(device)

                pred = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)
                y_true.extend(label.numpy().ravel().tolist())
                y_score.extend(torch.sigmoid(pred).cpu().numpy().ravel().tolist())

        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
        epoch_aucs.append(auc)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        print(f"[{epoch+1}/{epochs}] Loss={avg_loss:.4f}  Val AUC={auc:.4f}  Time={epoch_time:.2f}s")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{prefix}_best.pt")

    # 每轮记录
    auc_file = f"{prefix}_epoch_auc.txt"
    with open(auc_file, "w") as f:
        for i, (auc, t) in enumerate(zip(epoch_aucs, epoch_times)):
            f.write(f"{i+1}\t{auc:.6f}\t{t:.2f}\n")
    print(f"每轮AUC和训练时间已保存: {auc_file}")

    # ---------- 测试 ----------
    model.eval()
    y_true, y_score = [], []
    detail_rows = []  # 逐样本详细信息
    with torch.no_grad():
        for batch in test_dl:
            x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, metas = batch
            x1=x1.to(device); adj1=adj1.to(device); e1=e1.to(device)
            x2=x2.to(device); adj2=adj2.to(device); e2=e2.to(device)
            mask1=mask1.to(device); mask2=mask2.to(device)

            logits = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)    # [B,1]
            scores = torch.sigmoid(logits).cpu().numpy().ravel()
            preds  = (scores >= 0.5).astype(int)
            labels = label.numpy().ravel().astype(int)

            y_true.extend(labels.tolist())
            y_score.extend(scores.tolist())

            # 将每条样本写入列表
            for i, meta in enumerate(metas):
                detail_rows.append([
                    meta.get('target',''),
                    int(labels[i]),
                    int(preds[i]),
                    float(scores[i]),
                    meta.get('file',''),
                    meta.get('lig_file',''),
                    meta.get('pair_type','')
                ])

    # 测试集指标
    auc_test = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
    print(f"测试集AUC: {auc_test:.4f}")

    # ROC 存档
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_file = f"{prefix}_test_roc.csv"
    np.savetxt(roc_file, np.column_stack([fpr, tpr]), delimiter=',', header='FPR,TPR', comments='')
    print(f"ROC坐标已保存: {roc_file}")

    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(fpr, tpr, label=f'AUC={auc_test:.3f}')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    roc_img = f"{prefix}_test_roc.png"
    plt.savefig(roc_img, dpi=300, bbox_inches='tight'); plt.close()
    print(f"ROC曲线图片已保存: {roc_img}")

    # 逐样本详情 CSV
    detail_csv = f"{prefix}_test_details.csv"
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
    with open(detail_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "label", "pred_label", "score", "pair_file", "ligand_file", "pair_type"])
        writer.writerows(detail_rows)
    print(f"测试集逐样本详情已保存: {detail_csv}")


# ========= 4. 主控 =========
def main():
    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    EPOCHS = 20
    OUT_MODEL_PREFIX = 'gmn_edgeaware'
    ENCODE_ROOT = 'data/encode-LIT'
    MODE = 'edge_no_sidechain'

    # —— 注意：如果数据非常大，可以启用限流参数 —— #
    train, val, test = build_dataset(
        ENCODE_ROOT, MODE,
        split_ratio=(0.8, 0.1, 0.1),
        max_pairs_per_target=100,     # 例如 50000
        max_pos_per_ligand=100,       # 例如 200
        max_neg_per_ligand=100        # 例如 800
    )

    train_dl = DataLoader(PairGraphDataset(train), batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(PairGraphDataset(val), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl = DataLoader(PairGraphDataset(test), batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=True)

    # 维度从训练集第一条样本里读
    if len(train) == 0:
        raise RuntimeError("训练集为空，请检查数据目录与命名。")
    node_dim = train[0]['lig_node'].shape[-1]
    edge_dim = (train[0]['lig_edge'].shape[-1]) if train[0]['lig_edge'] is not None else 1

    model = GraphMatchingNetwork(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.5
    ).to(device)

    train_model(model, train_dl, val_dl, test_dl, device, OUT_MODEL_PREFIX, epochs=EPOCHS, lr=1e-3)


if __name__ == '__main__':
    main()
