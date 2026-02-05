# -*- coding: UTF-8 -*-
"""
Train GNN-MA on LIT-PCBA (encoded graphs)

Dataset layout (relative to this train.py file):
../data/encode-LIT/
  <TARGET_1>/
    edge_no_sidechain/
      ligand/   (*.npz)   # query ligands
      active/   (*.npz)   # positives (per target)
      decoy/    (*.npz)   # negatives (per target)
  <TARGET_2>/
    ...

Pairing rule (per target):
  For each ligand L in ligand/:
    (L, A) is a positive pair for every A in active/
    (L, D) is a negative pair for every D in decoy/

Outputs:
  ./run-LIT/<MODE>_<timestamp>/
    best_model.pt
    epoch_metrics.txt
    best_model_test_predictions.csv
"""
import os
import random
import math
import time
import csv
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from GNN_MA import GraphMatchingNetwork


# =========================
# Config (你主要改这里)
# =========================
MODE = "edge_no_sidechain"

BATCH_SIZE = 32
SAMPLES_PER_EPOCH = 200000       # 每个 epoch 跑多少样本（= steps * batch_size）
PRINT_EVERY = 4000

EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

WARMUP_EPOCHS = 2
LAMBDA_RANK_AFTER = 0.05
RANK_K_NEG = 10

# Hard negative mining（默认关闭）
USE_HARD_NEG = False
HARD_NEG_UPDATE_EVERY = 5
HARD_NEG_TOP_FRAC = 0.02
HARD_NEG_MAX_NEG_TO_SCORE = 100000

# 关键：按 target 组 batch（强烈推荐 EF）
USE_TARGET_BATCH = True
POS_IN_BATCH = 0.25              # 每个 batch 内 pos 比例（同一个 target 内）

# =========================
# Output / Saving
# =========================
BEST_BY = "ef"          # "ef" (Val EF@1%) or "auc" (Val AUC)
PRED_THRESHOLD = 0.5    # threshold for pred_label in CSV
RUNS_DIR = os.path.join(os.path.dirname(__file__), "run-LIT")  # ✅ LIT 输出目录


# =========================
# Utils
# =========================
def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def append_line(path: str, line: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def make_run_dir(mode: str) -> str:
    ensure_dir(RUNS_DIR)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = os.path.join(RUNS_DIR, f"{mode}_{run_id}")
    ensure_dir(run_dir)
    return run_dir


# =========================
# Dataset
# =========================
def _safe_list_npz(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")])


class PairGraphDataset(Dataset):
    """
    pair_list item format:
      {
        "target": str,
        "lig_file": str,   # ligand npz path
        "file": str,       # active/decoy npz path
        "label": int(0/1)
      }
    """
    def __init__(self, pair_list: List[Dict[str, Any]], enable_cache: bool = True, cache_max: int = 512):
        self.pair_list = pair_list
        for i, p in enumerate(self.pair_list):
            p["__idx"] = i

        # simple per-worker cache (helps a lot if ligand repeats across many pairs)
        self.enable_cache = bool(enable_cache)
        self.cache_max = int(cache_max)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.pair_list)

    def _load_npz_cached(self, path: str) -> Dict[str, np.ndarray]:
        if not self.enable_cache:
            with np.load(path) as dat:
                return {k: dat[k] for k in dat.files}

        if path in self._cache:
            return self._cache[path]

        with np.load(path) as dat:
            obj = {k: dat[k] for k in dat.files}

        # naive cache eviction
        if len(self._cache) >= self.cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = obj
        return obj

    def __getitem__(self, idx: int):
        p = self.pair_list[idx]

        lig_path = p["lig_file"]
        mol_path = p["file"]

        lig = self._load_npz_cached(lig_path)
        mol = self._load_npz_cached(mol_path)

        node1 = lig["node_feat"]
        adj1  = lig["adj"]
        e1    = lig["edge_feat"] if "edge_feat" in lig else adj1[..., None]

        node2 = mol["node_feat"]
        adj2  = mol["adj"]
        e2    = mol["edge_feat"] if "edge_feat" in mol else adj2[..., None]

        node1 = node1.astype(np.float32, copy=False)
        adj1  = adj1.astype(np.float32, copy=False)
        e1    = e1.astype(np.float32, copy=False)
        node2 = node2.astype(np.float32, copy=False)
        adj2  = adj2.astype(np.float32, copy=False)
        e2    = e2.astype(np.float32, copy=False)

        label = np.array([p["label"]], dtype=np.float32)
        meta  = {
            "idx": int(p["__idx"]),
            "target": p["target"],
            "lig_file": lig_path,
            "file": mol_path,
            "label": int(p["label"]),
        }
        return (node1, adj1, e1, node2, adj2, e2, label, meta)


def pad_collate(batch):
    max_n1 = max(x[0].shape[0] for x in batch)
    max_n2 = max(x[3].shape[0] for x in batch)

    xs1, adjs1, es1, xs2, adjs2, es2, masks1, masks2, labels, metas = [], [], [], [], [], [], [], [], [], []
    for x1, adj1, e1, x2, adj2, e2, label, meta in batch:
        n1, n2 = x1.shape[0], x2.shape[0]

        pad_x1   = np.pad(x1,  ((0, max_n1 - n1), (0, 0)), "constant")
        pad_x2   = np.pad(x2,  ((0, max_n2 - n2), (0, 0)), "constant")
        pad_adj1 = np.pad(adj1,((0, max_n1 - n1), (0, max_n1 - n1)), "constant")
        pad_adj2 = np.pad(adj2,((0, max_n2 - n2), (0, max_n2 - n2)), "constant")
        pad_e1   = np.pad(e1,  ((0, max_n1 - n1), (0, max_n1 - n1), (0, 0)), "constant")
        pad_e2   = np.pad(e2,  ((0, max_n2 - n2), (0, max_n2 - n2), (0, 0)), "constant")

        mask1 = np.zeros(max_n1, dtype=np.float32); mask1[:n1] = 1
        mask2 = np.zeros(max_n2, dtype=np.float32); mask2[:n2] = 1

        xs1.append(pad_x1); adjs1.append(pad_adj1); es1.append(pad_e1)
        xs2.append(pad_x2); adjs2.append(pad_adj2); es2.append(pad_e2)
        masks1.append(mask1); masks2.append(mask2)
        labels.append(label); metas.append(meta)

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


# =========================
# Build pairs (LIT-PCBA)
# =========================
def get_graph_pairs_from_target(target_dir: str, mode: str, target_name: str) -> List[Dict[str, Any]]:
    """
    Build Cartesian-product pairs inside one target:
      ligand x active -> label=1
      ligand x decoy  -> label=0
    """
    folder = os.path.join(target_dir, mode)

    lig_dir = os.path.join(folder, "ligand")
    act_dir = os.path.join(folder, "active")
    dec_dir = os.path.join(folder, "decoy")

    lig_files = _safe_list_npz(lig_dir)
    act_files = _safe_list_npz(act_dir)
    dec_files = _safe_list_npz(dec_dir)

    if (not lig_files) or (not act_files) or (not dec_files):
        return []

    all_pairs: List[Dict[str, Any]] = []
    for lf in lig_files:
        for af in act_files:
            all_pairs.append({"target": target_name, "lig_file": lf, "file": af, "label": 1})
        for df in dec_files:
            all_pairs.append({"target": target_name, "lig_file": lf, "file": df, "label": 0})

    random.shuffle(all_pairs)
    return all_pairs


def build_dataset(encode_root: str, mode: str, split_ratio=(0.8, 0.1, 0.1)):
    all_targets = [d for d in os.listdir(encode_root) if os.path.isdir(os.path.join(encode_root, d))]
    all_pairs: List[Dict[str, Any]] = []

    for t in sorted(all_targets):
        target_dir = os.path.join(encode_root, t)
        if not os.path.exists(os.path.join(target_dir, mode)):
            continue
        pairs_t = get_graph_pairs_from_target(target_dir, mode, target_name=t)
        if pairs_t:
            all_pairs.extend(pairs_t)

    random.shuffle(all_pairs)
    n_total = len(all_pairs)
    if n_total == 0:
        raise RuntimeError(f"No pairs built. Please check encode_root={encode_root} and MODE={mode} structure.")

    n_train = int(n_total * split_ratio[0])
    n_val   = int(n_total * split_ratio[1])

    train = all_pairs[:n_train]
    val   = all_pairs[n_train:n_train+n_val]
    test  = all_pairs[n_train+n_val:]

    def _count(pairs):
        y = np.array([p["label"] for p in pairs], dtype=np.int64)
        return int(y.sum()), int((1-y).sum())

    tr_p, tr_n = _count(train)
    va_p, va_n = _count(val)
    te_p, te_n = _count(test)
    print(f"Total pair count: {n_total}", flush=True)
    print(f"Train: pos={tr_p} neg={tr_n}  pos%={tr_p/(tr_p+tr_n+1e-9):.4f}", flush=True)
    print(f"Val  : pos={va_p} neg={va_n}  pos%={va_p/(va_p+va_n+1e-9):.4f}", flush=True)
    print(f"Test : pos={te_p} neg={te_n}  pos%={te_p/(te_p+te_n+1e-9):.4f}", flush=True)
    return train, val, test


# =========================
# EF / AUC
# =========================
def compute_ef(scores, labels, top_frac=0.01):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    if n == 0:
        return float("nan")
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    top_n = max(1, int(math.ceil(top_frac * n)))
    order = np.argsort(scores)[::-1]
    hits = int(labels[order[:top_n]].sum())
    return (hits / top_n) / (n_pos / n)


@torch.no_grad()
def evaluate_auc_ef(dl, model, device, top_frac=0.01):
    model.eval()
    y_true, y_score, y_target = [], [], []

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for batch in dl:
            x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, metas = batch
            x1 = x1.to(device); adj1 = adj1.to(device); e1 = e1.to(device)
            x2 = x2.to(device); adj2 = adj2.to(device); e2 = e2.to(device)
            mask1 = mask1.to(device); mask2 = mask2.to(device)

            logits = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)
            logits = logits.view(-1).detach().cpu().numpy()
            scores = sigmoid_np(logits)
            labels = label.view(-1).numpy().astype(int)

            y_true.extend(labels.tolist())
            y_score.extend(scores.tolist())
            y_target.extend([m["target"] for m in metas])

    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5

    per_t = {}
    for t, lab, sc in zip(y_target, y_true, y_score):
        per_t.setdefault(t, {"y": [], "s": []})
        per_t[t]["y"].append(lab)
        per_t[t]["s"].append(sc)

    ef_list = []
    for d in per_t.values():
        ef = compute_ef(d["s"], d["y"], top_frac=top_frac)
        if not math.isnan(ef):
            ef_list.append(ef)
    ef_mean = float(np.mean(ef_list)) if ef_list else float("nan")

    return auc, ef_mean, n_pos, n_neg


@torch.no_grad()
def predict_records(dl, model, device, threshold: float = 0.5):
    model.eval()
    records = []

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for batch in dl:
            x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, metas = batch
            x1 = x1.to(device); adj1 = adj1.to(device); e1 = e1.to(device)
            x2 = x2.to(device); adj2 = adj2.to(device); e2 = e2.to(device)
            mask1 = mask1.to(device); mask2 = mask2.to(device)

            logits = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)
            logits = logits.view(-1).detach().cpu().numpy()
            scores = sigmoid_np(logits)
            labels = label.view(-1).numpy().astype(int)

            for i, m in enumerate(metas):
                sc = float(scores[i])
                lab = int(labels[i])
                pred = int(sc >= float(threshold))
                records.append({
                    "idx": int(m.get("idx", -1)),
                    "target": m.get("target", ""),
                    "lig_file": m.get("lig_file", ""),
                    "pair_file": m.get("file", ""),
                    "label": lab,
                    "score": sc,
                    "pred_label": pred,
                    "test_result": int(pred == lab),
                })

    return records


def save_predictions_csv(records, out_path: str) -> None:
    if not records:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)


# =========================
# Ranking loss
# =========================
def batch_pairwise_rank_loss(logits_1d, labels_1d, metas, k_neg=10):
    device = logits_1d.device
    targets = [m["target"] for m in metas]
    uniq = list(set(targets))

    losses = []
    for t in uniq:
        idx = [i for i, tt in enumerate(targets) if tt == t]
        if len(idx) < 2:
            continue
        idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
        s_t = logits_1d.index_select(0, idx_t)
        y_t = labels_1d.index_select(0, idx_t)

        pos_mask = y_t > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        s_pos = s_t[pos_mask]
        s_neg = s_t[neg_mask]
        k = min(k_neg, s_neg.numel())
        s_neg, _ = torch.topk(s_neg, k=k, largest=True, sorted=False)

        diff = s_pos[:, None] - s_neg[None, :]
        losses.append(F.softplus(-diff).mean())

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# =========================
# Target batch sampler
# =========================
class TargetBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, pairs: List[Dict[str, Any]], batch_size: int, pos_in_batch: float, samples_per_epoch: int):
        self.batch_size = int(batch_size)
        self.pos_in_batch = float(pos_in_batch)
        self.samples_per_epoch = int(samples_per_epoch)

        self.t_pos, self.t_neg = {}, {}
        for i, p in enumerate(pairs):
            t = p["target"]
            if int(p["label"]) == 1:
                self.t_pos.setdefault(t, []).append(i)
            else:
                self.t_neg.setdefault(t, []).append(i)

        self.targets = [t for t in self.t_pos.keys() if t in self.t_neg and len(self.t_pos[t]) > 0 and len(self.t_neg[t]) > 0]
        if not self.targets:
            raise RuntimeError("No valid targets with BOTH pos and neg found. Check your dataset.")

        w = [math.sqrt(len(self.t_pos[t]) + 1e-9) for t in self.targets]
        s = sum(w)
        self.t_weights = [x / s for x in w]

        self.steps = int(math.ceil(self.samples_per_epoch / self.batch_size))

    def __len__(self):
        return self.steps

    def __iter__(self):
        n_pos = max(1, int(round(self.batch_size * self.pos_in_batch)))
        n_neg = self.batch_size - n_pos
        for _ in range(self.steps):
            t = random.choices(self.targets, weights=self.t_weights, k=1)[0]
            pos_pool = self.t_pos[t]
            neg_pool = self.t_neg[t]
            pos_idx = random.choices(pos_pool, k=n_pos)
            neg_idx = random.choices(neg_pool, k=n_neg)
            batch = pos_idx + neg_idx
            random.shuffle(batch)
            yield batch


# =========================
# Hard neg mining
# =========================
@torch.no_grad()
def mine_hard_negative_scores(model, train_ds: PairGraphDataset, device, mining_dl_kwargs, top_frac=0.02, max_neg_to_score=100000):
    neg_indices = [i for i, p in enumerate(train_ds.pair_list) if int(p["label"]) == 0]
    if not neg_indices:
        return np.zeros(len(train_ds), dtype=np.float32)

    if max_neg_to_score is not None and len(neg_indices) > max_neg_to_score:
        neg_indices = random.sample(neg_indices, max_neg_to_score)

    subset = Subset(train_ds, neg_indices)
    dl = DataLoader(subset, **mining_dl_kwargs)

    scores = np.zeros(len(train_ds), dtype=np.float32)
    model.eval()

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for batch in dl:
            x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, metas = batch
            x1 = x1.to(device); adj1 = adj1.to(device); e1 = e1.to(device)
            x2 = x2.to(device); adj2 = adj2.to(device); e2 = e2.to(device)
            mask1 = mask1.to(device); mask2 = mask2.to(device)

            logits = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)
            prob = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()

            for i, m in enumerate(metas):
                scores[int(m["idx"])] = float(prob[i])

    mined = scores[neg_indices]
    if mined.size > 0:
        thr = np.quantile(mined, 1.0 - top_frac)
        scores = scores * (scores >= thr).astype(np.float32)

    return scores


# =========================
# Train
# =========================
def train_model(model, train_pairs, val_pairs, test_pairs, device, run_dir: str):
    train_ds = PairGraphDataset(train_pairs, enable_cache=True, cache_max=2048)
    val_ds   = PairGraphDataset(val_pairs, enable_cache=False)
    test_ds  = PairGraphDataset(test_pairs, enable_cache=False)

    ensure_dir(run_dir)
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")
    epoch_log_path = os.path.join(run_dir, "epoch_metrics.txt")
    best_test_csv_path = os.path.join(run_dir, "best_model_test_predictions.csv")

    append_line(epoch_log_path, f"# start_time: {now_str()}")
    append_line(epoch_log_path, f"# run_dir: {run_dir}")
    append_line(epoch_log_path, f"# MODE={MODE}  EPOCHS={EPOCHS}  BATCH_SIZE={BATCH_SIZE}  LR={LR}  WEIGHT_DECAY={WEIGHT_DECAY}  DROPOUT={DROPOUT}")
    append_line(epoch_log_path, f"# BEST_BY={BEST_BY}  PRED_THRESHOLD={PRED_THRESHOLD}")
    append_line(epoch_log_path, "timestamp\tepoch\tloss\tbce\trank\tlambda\tval_auc\tval_ef1\tval_pos\tval_neg\twarmup\tepoch_time_s\tis_best")

    num_workers = 0 if os.name == "nt" else 4
    dl_kwargs = dict(collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    val_dl  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs)

    samples_per_epoch = int(SAMPLES_PER_EPOCH) if SAMPLES_PER_EPOCH is not None else len(train_ds)

    if USE_TARGET_BATCH:
        batch_sampler = TargetBatchSampler(train_pairs, BATCH_SIZE, POS_IN_BATCH, samples_per_epoch)
        train_dl = DataLoader(train_ds, batch_sampler=batch_sampler, shuffle=False, **dl_kwargs)
    else:
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **dl_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    bce_loss_fn = nn.BCEWithLogitsLoss()

    mining_dl_kwargs = dict(dl_kwargs)
    mining_dl_kwargs.update(dict(batch_size=max(64, BATCH_SIZE), shuffle=False))

    best_metric = -1e18
    best_epoch = -1

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()

        in_warmup = epoch < WARMUP_EPOCHS
        lambda_rank = 0.0 if in_warmup else float(LAMBDA_RANK_AFTER)
        enable_rank = lambda_rank > 0

        losses, bce_losses, rank_losses = [], [], []
        seen, step_t0 = 0, time.time()

        for step, batch in enumerate(train_dl, start=1):
            x1, adj1, e1, x2, adj2, e2, mask1, mask2, label, metas = batch
            x1 = x1.to(device); adj1 = adj1.to(device); e1 = e1.to(device)
            x2 = x2.to(device); adj2 = adj2.to(device); e2 = e2.to(device)
            mask1 = mask1.to(device); mask2 = mask2.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x1, adj1, e1, x2, adj2, e2, mask1, mask2)

                logits_1d = logits.view(-1)
                label_1d  = label.view(-1)

                loss_bce = bce_loss_fn(logits_1d, label_1d)

                if enable_rank:
                    loss_rank = batch_pairwise_rank_loss(logits_1d, label_1d, metas, k_neg=RANK_K_NEG)
                    loss = loss_bce + lambda_rank * loss_rank
                else:
                    loss_rank = torch.tensor(0.0, device=device)
                    loss = loss_bce

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))
            bce_losses.append(float(loss_bce.detach().cpu().item()))
            rank_losses.append(float(loss_rank.detach().cpu().item()))
            seen += int(label_1d.numel())

            if PRINT_EVERY and (step % PRINT_EVERY == 0):
                dt = time.time() - step_t0
                sps = seen / max(dt, 1e-9)
                with torch.no_grad():
                    prob = torch.sigmoid(logits_1d.detach())
                    pos_mask = label_1d > 0.5
                    neg_mask = ~pos_mask
                    pos_mean = float(prob[pos_mask].mean().item()) if pos_mask.any() else float("nan")
                    neg_mean = float(prob[neg_mask].mean().item()) if neg_mask.any() else float("nan")
                    pos_frac = float(pos_mask.float().mean().item())

                print(
                    f"  step {step:6d}/{len(train_dl):6d} | "
                    f"avg_loss={np.mean(losses):.4f} (bce={np.mean(bce_losses):.4f}, rank={np.mean(rank_losses):.4f}, λ={lambda_rank:.3f}) | "
                    f"{sps:.1f} samples/s | batch_pos={pos_frac:.2f} p(pos)={pos_mean:.3f} p(neg)={neg_mean:.3f}",
                    flush=True
                )

        avg_loss = float(np.mean(losses))
        avg_bce  = float(np.mean(bce_losses))
        avg_rank = float(np.mean(rank_losses))

        auc_val, ef_val, vpos, vneg = evaluate_auc_ef(val_dl, model, device, top_frac=0.01)

        t_ep = time.time() - t0
        ts = now_str()

        best_by = str(BEST_BY).strip().lower()
        if best_by == "auc":
            metric_val = float(auc_val)
            metric_name = "val_auc"
        else:
            metric_name = "val_ef1"
            if ef_val is None or (isinstance(ef_val, float) and math.isnan(ef_val)):
                metric_val = float(auc_val)
                metric_name = "val_auc_fallback"
            else:
                metric_val = float(ef_val)

        is_best = False
        if metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch + 1
            is_best = True
            ckpt = {
                "epoch": best_epoch,
                "best_metric_name": metric_name,
                "best_metric": float(best_metric),
                "val_auc": float(auc_val),
                "val_ef1": float(ef_val) if not (isinstance(ef_val, float) and math.isnan(ef_val)) else float("nan"),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(ckpt, os.path.join(run_dir, "best_model.pt"))

        print(
            f"{ts} [{epoch+1}/{EPOCHS}] Loss={avg_loss:.4f} (BCE={avg_bce:.4f}, Rank={avg_rank:.4f}, λ={lambda_rank:.3f})  "
            f"Val AUC={auc_val:.4f} Val EF@1%={ef_val:.3f}  (val_pos={vpos}, val_neg={vneg})  "
            f"warmup={in_warmup}  Time={t_ep:.2f}s" +
            (f"  ✅BEST({metric_name}={best_metric:.4f})" if is_best else ""),
            flush=True
        )

        append_line(
            epoch_log_path,
            f"{ts}\t{epoch+1}\t{avg_loss:.6f}\t{avg_bce:.6f}\t{avg_rank:.6f}\t{lambda_rank:.3f}\t{auc_val:.6f}\t{ef_val:.6f}\t{vpos}\t{vneg}\t{int(in_warmup)}\t{t_ep:.2f}\t{int(is_best)}"
        )

    # -------- best model test + save CSV --------
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            best_epoch = int(ckpt.get("epoch", best_epoch))
        else:
            model.load_state_dict(ckpt)

    test_ds  = PairGraphDataset(test_pairs, enable_cache=False)
    num_workers = 0 if os.name == "nt" else 4
    dl_kwargs = dict(collate_fn=pad_collate, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs)

    auc_test, ef_test, tpos, tneg = evaluate_auc_ef(test_dl, model, device, top_frac=0.01)
    ts = now_str()
    print(f"{ts} [BEST TEST @ epoch {best_epoch}] AUC={auc_test:.4f} EF@1%={ef_test:.3f} (test_pos={tpos}, test_neg={tneg})", flush=True)

    records = predict_records(test_dl, model, device, threshold=float(PRED_THRESHOLD))
    save_predictions_csv(records, os.path.join(run_dir, "best_model_test_predictions.csv"))


def infer_dims_from_one_pair(pairs: List[Dict[str, Any]]) -> Tuple[int, int]:
    p0 = pairs[0]
    with np.load(p0["lig_file"]) as dat:
        node_dim = int(dat["node_feat"].shape[-1])
        if "edge_feat" in dat:
            edge_dim = int(dat["edge_feat"].shape[-1])
        else:
            edge_dim = 1
    return node_dim, edge_dim


def main():
    set_seed(2025)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    run_dir = make_run_dir(MODE)
    print("Run outputs will be saved to:", run_dir, flush=True)

    encode_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "encode-LIT"))
    if not os.path.isdir(encode_root):
        raise FileNotFoundError(f"ENCODE_ROOT not found: {encode_root}")

    train_pairs, val_pairs, test_pairs = build_dataset(encode_root, MODE)

    node_dim, edge_dim = infer_dims_from_one_pair(train_pairs)

    model = GraphMatchingNetwork(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=64,
        num_layers=3,
        dropout=DROPOUT,
    ).to(device)

    train_model(model, train_pairs, val_pairs, test_pairs, device, run_dir=run_dir)


if __name__ == "__main__":
    main()
