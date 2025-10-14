# -*- coding: UTF-8 -*-
'''
@Author  ：Iris
@Date    ：2025-09-28 18:45 
'''
# -*- coding: UTF-8 -*-

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


# ========= 0. Small Tools =========
def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ========= 1. Dataset and Data Loading =========
class PairGraphDataset(Dataset):
    """
    The pair is structured as a dict:
      {
        'target':  target name,
        'file':    path to another molecule's .npz file,
        'label':   0/1,
        'lig_node','lig_adj','lig_edge': crystal ligand (left figure)
      }
    """
    def __init__(self, pair_list):
        self.pair_list = pair_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        p = self.pair_list[idx]
        lig_node = p['lig_node']; lig_adj = p['lig_adj']; lig_edge = p['lig_edge']
        node1 = lig_node.astype(np.float32, copy=False)
        adj1  = lig_adj.astype(np.float32, copy=False)
        e1    = lig_edge.astype(np.float32, copy=False) if lig_edge is not None else adj1[..., None]

        # Read another molecule
        dat = np.load(p['file'])  #  If faster I/O is needed, can be changed to mmap_mode='r'
        node2 = dat['node_feat'].astype(np.float32, copy=False)
        adj2  = dat['adj'].astype(np.float32, copy=False)
        e2    = dat['edge_feat'].astype(np.float32, copy=False) if 'edge_feat' in dat else adj2[..., None]

        label = np.array([p['label']], dtype=np.float32)
        meta = {'target': p['target'], 'file': p['file'], 'label': int(p['label'])}
        return (node1, adj1, e1, node2, adj2, e2, label, meta)


def pad_collate(batch):
    # Calculate pad size
    max_n1 = max(x[0].shape[0] for x in batch)
    max_n2 = max(x[3].shape[0] for x in batch)

    xs1, adjs1, es1, xs2, adjs2, es2, masks1, masks2, labels, metas = [], [], [], [], [], [], [], [], [], []
    for x1, adj1, e1, x2, adj2, e2, label, meta in batch:
        n1, n2 = x1.shape[0], x2.shape[0]
        # Node features
        pad_x1 = np.pad(x1, ((0, max_n1-n1), (0, 0)), 'constant')
        pad_x2 = np.pad(x2, ((0, max_n2-n2), (0, 0)), 'constant')
        # Adjacency
        pad_adj1 = np.pad(adj1, ((0, max_n1-n1), (0, max_n1-n1)), 'constant')
        pad_adj2 = np.pad(adj2, ((0, max_n2-n2), (0, max_n2-n2)), 'constant')
        # Edge features [N,N,de]
        pad_e1 = np.pad(e1, ((0, max_n1-n1), (0, max_n1-n1), (0, 0)), 'constant')
        pad_e2 = np.pad(e2, ((0, max_n2-n2), (0, max_n2-n2), (0, 0)), 'constant')

        mask1 = np.zeros(max_n1, dtype=np.float32); mask1[:n1] = 1
        mask2 = np.zeros(max_n2, dtype=np.float32); mask2[:n2] = 1

        xs1.append(pad_x1); adjs1.append(pad_adj1); es1.append(pad_e1)
        xs2.append(pad_x2); adjs2.append(pad_adj2); es2.append(pad_e2)
        masks1.append(mask1); masks2.append(mask2)
        labels.append(label); metas.append(meta)

    # Use numpy.stack + torch.from_numpy (more memory-efficient for copying)
    return (
        torch.from_numpy(np.stack(xs1,  axis=0)).float(),
        torch.from_numpy(np.stack(adjs1,axis=0)).float(),
        torch.from_numpy(np.stack(es1,  axis=0)).float(),
        torch.from_numpy(np.stack(xs2,  axis=0)).float(),
        torch.from_numpy(np.stack(adjs2,axis=0)).float(),
        torch.from_numpy(np.stack(es2,  axis=0)).float(),
        torch.from_numpy(np.stack(masks1,axis=0)).float(),
        torch.from_numpy(np.stack(masks2,axis=0)).float(),
        torch.from_numpy(np.stack(labels,axis=0)).float(),
        metas
    )


# ========= 2. Pair Collection and division =========
def get_graph_pairs_from_target(target_dir, mode, target_name):
    folder = os.path.join(target_dir, mode)
    ligand_file = os.path.join(folder, 'crystal_ligand.npz')
    ligand = np.load(ligand_file)
    lig_node, lig_adj = ligand['node_feat'], ligand['adj']
    lig_edge = ligand['edge_feat'] if 'edge_feat' in ligand else None

    act_dir = os.path.join(folder, 'active')
    dec_dir = os.path.join(folder, 'decoy')
    act_files = sorted([os.path.join(act_dir, f) for f in os.listdir(act_dir) if f.endswith('.npz')]) if os.path.exists(act_dir) else []
    dec_files = sorted([os.path.join(dec_dir, f) for f in os.listdir(dec_dir) if f.endswith('.npz')]) if os.path.exists(dec_dir) else []

    pos_pairs = [{
        'target': target_name, 'file': f, 'label': 1,
        'lig_node': lig_node, 'lig_adj': lig_adj, 'lig_edge': lig_edge
    } for f in act_files]
    neg_pairs = [{
        'target': target_name, 'file': f, 'label': 0,
        'lig_node': lig_node, 'lig_adj': lig_adj, 'lig_edge': lig_edge
    } for f in dec_files]
    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    return all_pairs


def build_dataset(encode_root, mode, split_ratio=(0.8,0.1,0.1), max_per_target=None):
    all_targets = [d for d in os.listdir(encode_root) if os.path.isdir(os.path.join(encode_root, d))]
    all_pairs = []
    for t in all_targets:
        target_dir = os.path.join(encode_root, t)
        if not os.path.exists(os.path.join(target_dir, mode)):
            continue
        pairs = get_graph_pairs_from_target(target_dir, mode, target_name=t)
        if max_per_target:
            random.shuffle(pairs); pairs = pairs[:max_per_target]
        all_pairs.extend(pairs)

    print(f"Total pair count: {len(all_pairs)}")
    random.shuffle(all_pairs)
    n_total = len(all_pairs)
    n_train = int(n_total * split_ratio[0])
    n_val   = int(n_total * split_ratio[1])
    train = all_pairs[:n_train]
    val   = all_pairs[n_train:n_train+n_val]
    test  = all_pairs[n_train+n_val:]
    return train, val, test


# ========= 3. Training loop =========
def train_model(model, train_dl, val_dl, test_dl, device, out_model_prefix, epochs=20, lr=1e-3):
    save_dir = "save_data/edge_no_sidechain_cross_double-2"
    os.makedirs(save_dir, exist_ok=True)
    prefix = os.path.join(save_dir, os.path.basename(out_model_prefix))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_auc = 0.0
    epoch_aucs, epoch_times = [], []

    # ---------- Training ----------
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        losses = []
        for batch in train_dl:
            x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, _ = batch
            x1=x1.to(device, non_blocking=True); adj1=adj1.to(device, non_blocking=True); e1=e1.to(device, non_blocking=True)
            x2=x2.to(device, non_blocking=True); adj2=adj2.to(device, non_blocking=True); e2=e2.to(device, non_blocking=True)
            mask1=mask1.to(device, non_blocking=True); mask2=mask2.to(device, non_blocking=True); label=label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                pred = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)
                loss = F.binary_cross_entropy_with_logits(pred, label)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # ---------- Validation ----------
        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                for batch in val_dl:
                    x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, _ = batch
                    x1=x1.to(device, non_blocking=True); adj1=adj1.to(device, non_blocking=True); e1=e1.to(device, non_blocking=True)
                    x2=x2.to(device, non_blocking=True); adj2=adj2.to(device, non_blocking=True); e2=e2.to(device, non_blocking=True)
                    mask1=mask1.to(device, non_blocking=True); mask2=mask2.to(device, non_blocking=True)

                    pred = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)
                    # The label only needs to stay on the CPU
                    y_true.extend(label.numpy().ravel().tolist())
                    y_score.extend(torch.sigmoid(pred).cpu().numpy().ravel().tolist())

        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
        epoch_aucs.append(auc)
        t1 = time.time()
        epoch_time = t1 - t0
        epoch_times.append(epoch_time)
        print(f"[{epoch+1}/{epochs}] Loss={avg_loss:.4f}  Val AUC={auc:.4f}  Time={epoch_time:.2f}s")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{prefix}_best.pt")

    # Record per round
    auc_file = f"{prefix}_epoch_auc.txt"
    with open(auc_file, "w") as f:
        for i, (auc, t) in enumerate(zip(epoch_aucs, epoch_times)):
            f.write(f"{i+1}\t{auc:.6f}\t{t:.2f}\n")
    print(f"The AUC and training time per round have been saved: {auc_file}")

    # ---------- Test ----------
    model.eval()
    y_true, y_score = [], []
    detail_rows = []  # Sample-by-sample detailed information
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            for batch in test_dl:
                x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, metas = batch
                x1=x1.to(device, non_blocking=True); adj1=adj1.to(device, non_blocking=True); e1=e1.to(device, non_blocking=True)
                x2=x2.to(device, non_blocking=True); adj2=adj2.to(device, non_blocking=True); e2=e2.to(device, non_blocking=True)
                mask1=mask1.to(device, non_blocking=True); mask2=mask2.to(device, non_blocking=True)

                logits = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)    # [B,1]
                scores = torch.sigmoid(logits).cpu().numpy().ravel()
                preds  = (scores >= 0.5).astype(int)
                labels = label.numpy().ravel().astype(int)

                y_true.extend(labels.tolist())
                y_score.extend(scores.tolist())

                for i, meta in enumerate(metas):
                    detail_rows.append([
                        meta['target'],
                        int(labels[i]),
                        int(preds[i]),
                        float(scores[i]),
                        meta['file']
                    ])

    # Test set metrics
    auc_test = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
    print(f"Test set AUC: {auc_test:.4f}")

    # ROC archiving
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_file = f"{prefix}_test_roc.csv"
    np.savetxt(roc_file, np.column_stack([fpr, tpr]), delimiter=',', header='FPR,TPR', comments='')
    print(f"ROC coordinates saved: {roc_file}")

    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(fpr, tpr, label=f'AUC={auc_test:.3f}')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    roc_img = f"{prefix}_test_roc.png"
    plt.savefig(roc_img, dpi=300, bbox_inches='tight'); plt.close()
    print(f"ROC curve image saved: {roc_img}")

    # Sample-by-sample details CSV
    detail_csv = f"{prefix}_test_details.csv"
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
    with open(detail_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "label", "pred_label", "score", "file"])
        writer.writerows(detail_rows)
    print(f"Test set sample-by-sample details saved: {detail_csv}")


# ========= 4. Main control =========
def main():
    set_seed(2025)

    # —— cuDNN / Matmul optimization ——
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')  # PyTorch>=2.0
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    EPOCHS = 20
    OUT_MODEL_PREFIX = 'gmn_edgeaware'
    ENCODE_ROOT = 'data/encode-DUD-E'
    MODE = 'edge_no_sidechain'

    train, val, test = build_dataset(ENCODE_ROOT, MODE)

    # —— DataLoader Tuning Parameters ——
    common_kwargs = dict(
        batch_size=BATCH_SIZE,
        collate_fn=pad_collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    train_dl = DataLoader(PairGraphDataset(train), shuffle=True,  **common_kwargs)
    val_dl   = DataLoader(PairGraphDataset(val),   shuffle=False, **common_kwargs)
    test_dl  = DataLoader(PairGraphDataset(test),  shuffle=False, **common_kwargs)

    # Read dimensions from the first sample of the training set
    node_dim = train[0]['lig_node'].shape[-1]
    edge_dim = (train[0]['lig_edge'].shape[-1]) if train[0]['lig_edge'] is not None else 1

    model = GraphMatchingNetwork(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.5
    ).to(device)

    # (Optional) If you are using PyTorch 2.x and want to squeeze out a bit more speed, you can turn on:
    # try:
    #     if torch.__version__ >= '2.0':
    #         model = torch.compile(model)
    # except Exception:
    #     pass

    train_model(model, train_dl, val_dl, test_dl, device, OUT_MODEL_PREFIX, epochs=EPOCHS, lr=1e-3)


if __name__ == '__main__':
    main()
