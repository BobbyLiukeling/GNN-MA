# -*- coding: utf-8 -*-
import json
import random
from pathlib import Path
from typing import List, Tuple

# =========================
# 配置：LIT 根目录
# =========================
ROOT = Path(r"E:\code\python\daily\Alignment\data\encode-LIT")
EDGE_DIR_NAME = "edge_no_sidechain"

# 8:1:1
RATIO_TRAIN = 0.8
RATIO_VAL   = 0.1
RATIO_TEST  = 0.1
SEED = 20260227

# 编码文件后缀
ALLOW_EXTS = {".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".bin"}

# ===== 路径/文件名关键字识别正负
POS_KEYS = ("active", "actives", "pos", "positive")
NEG_KEYS = ("inactive", "inactives", "neg", "negative", "decoy", "decoys")


def _is_pos_path(p: Path) -> bool:
    s = str(p).lower()
    return any(k in s for k in POS_KEYS)


def _is_neg_path(p: Path) -> bool:
    s = str(p).lower()
    return any(k in s for k in NEG_KEYS)


def scan_candidates(edge_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """返回：(pos_files, neg_files, unknown_files)"""
    pos, neg, unk = [], [], []
    for fp in edge_dir.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in ALLOW_EXTS:
            continue
        if _is_pos_path(fp):
            pos.append(fp)
        elif _is_neg_path(fp):
            neg.append(fp)
        else:
            unk.append(fp)
    return pos, neg, unk


def split_list(items: List[Path], seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """对 items 做 8/1/1 划分"""
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * RATIO_TRAIN)
    n_val   = int(n * RATIO_VAL)
    train = items[:n_train]
    val   = items[n_train:n_train + n_val]
    test  = items[n_train + n_val:]

    # 容错：避免 val/test 为空（n>=3 时）
    if n >= 3:
        if len(val) == 0:
            val = [train.pop()] if train else [test.pop()]
        if len(test) == 0:
            test = [train.pop()] if train else [val.pop()]

    return train, val, test


def write_rel_list(paths: List[Path], out_txt: Path, base: Path):
    """写相对路径（相对 edge_dir）"""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.relative_to(base)).replace("\\", "/") + "\n")


def overlap_count(a: List[Path], b: List[Path]) -> int:
    return len(set(map(str, a)) & set(map(str, b)))


def process_one_target(target_dir: Path) -> dict:
    edge_dir = target_dir / EDGE_DIR_NAME
    target = target_dir.name

    pos_files, neg_files, unk_files = scan_candidates(edge_dir)

    # 无法识别正负 -> 跳过（避免错误划分）
    if len(pos_files) == 0 or len(neg_files) == 0:
        return {
            "target": target,
            "status": "SKIP",
            "reason": "cannot_identify_pos_neg_by_path_keywords",
            "pos_total": len(pos_files),
            "neg_total": len(neg_files),
            "unknown_total": len(unk_files),
            "train_total": 0, "val_total": 0, "test_total": 0,
            "ov_train_val": -1, "ov_train_test": -1, "ov_val_test": -1
        }

    # 分层切分（pos/neg 各自 8:1:1）
    base_seed = SEED + (abs(hash(target)) % 100000)
    pos_tr, pos_va, pos_te = split_list(pos_files, seed=base_seed + 1)
    neg_tr, neg_va, neg_te = split_list(neg_files, seed=base_seed + 2)

    train = pos_tr + neg_tr
    val   = pos_va + neg_va
    test  = pos_te + neg_te

    ov_tr_va = overlap_count(train, val)
    ov_tr_te = overlap_count(train, test)
    ov_va_te = overlap_count(val, test)

    out_root = target_dir / "split_811"
    out_root.mkdir(parents=True, exist_ok=True)

    # 写出清单（相对 edge_dir）
    write_rel_list(pos_tr, out_root / "actives_train.txt", edge_dir)
    write_rel_list(pos_va, out_root / "actives_val.txt",   edge_dir)
    write_rel_list(pos_te, out_root / "actives_test.txt",  edge_dir)

    write_rel_list(neg_tr, out_root / "inactives_train.txt", edge_dir)
    write_rel_list(neg_va, out_root / "inactives_val.txt",   edge_dir)
    write_rel_list(neg_te, out_root / "inactives_test.txt",  edge_dir)

    write_rel_list(train, out_root / "candidates_train.txt", edge_dir)
    write_rel_list(val,   out_root / "candidates_val.txt",   edge_dir)
    write_rel_list(test,  out_root / "candidates_test.txt",  edge_dir)

    meta = {
        "target": target,
        "edge_dir": str(edge_dir),
        "seed": SEED,
        "ratio": {"train": RATIO_TRAIN, "val": RATIO_VAL, "test": RATIO_TEST},
        "counts": {
            "pos_total": len(pos_files),
            "neg_total": len(neg_files),
            "unknown_total": len(unk_files),
            "train_pos": len(pos_tr), "train_neg": len(neg_tr), "train_total": len(train),
            "val_pos": len(pos_va),   "val_neg": len(neg_va),   "val_total": len(val),
            "test_pos": len(pos_te),  "test_neg": len(neg_te),  "test_total": len(test),
        },
        "overlap": {
            "train_vs_val": ov_tr_va,
            "train_vs_test": ov_tr_te,
            "val_vs_test": ov_va_te
        },
        "note": "Candidate molecule-disjoint stratified split (8:1:1). Query ligands may be reused across splits; candidates are disjoint.",
        "pos_keys": list(POS_KEYS),
        "neg_keys": list(NEG_KEYS),
        "allow_exts": sorted(list(ALLOW_EXTS)),
    }

    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    status = "OK" if (ov_tr_va == 0 and ov_tr_te == 0 and ov_va_te == 0) else "OVERLAP"

    return {
        "target": target,
        "status": status,
        "reason": "",
        "pos_total": len(pos_files),
        "neg_total": len(neg_files),
        "unknown_total": len(unk_files),
        "train_total": len(train),
        "val_total": len(val),
        "test_total": len(test),
        "ov_train_val": ov_tr_va,
        "ov_train_test": ov_tr_te,
        "ov_val_test": ov_va_te
    }


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT not found: {ROOT}")

    target_dirs = []
    for p in ROOT.iterdir():
        if not p.is_dir():
            continue
        if (p / EDGE_DIR_NAME).exists():
            target_dirs.append(p)

    target_dirs = sorted(target_dirs, key=lambda x: x.name)
    print(f"[INFO] Found {len(target_dirs)} targets under {ROOT} with '{EDGE_DIR_NAME}'")

    results = []
    for i, tdir in enumerate(target_dirs, 1):
        print(f"[{i}/{len(target_dirs)}] Processing target: {tdir.name}")
        try:
            info = process_one_target(tdir)
        except Exception as e:
            info = {
                "target": tdir.name,
                "status": "ERROR",
                "reason": repr(e),
                "pos_total": 0, "neg_total": 0, "unknown_total": 0,
                "train_total": 0, "val_total": 0, "test_total": 0,
                "ov_train_val": -1, "ov_train_test": -1, "ov_val_test": -1
            }
        results.append(info)

    report_path = ROOT / "split_811_overlap_report.csv"
    header = (
        "target,status,reason,pos_total,neg_total,unknown_total,"
        "train_total,val_total,test_total,ov_train_val,ov_train_test,ov_val_test\n"
    )
    with report_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for r in results:
            f.write(
                f"{r['target']},{r['status']},{r['reason']},"
                f"{r['pos_total']},{r['neg_total']},{r['unknown_total']},"
                f"{r['train_total']},{r['val_total']},{r['test_total']},"
                f"{r['ov_train_val']},{r['ov_train_test']},{r['ov_val_test']}\n"
            )

    ok = sum(1 for r in results if r["status"] == "OK")
    skip = sum(1 for r in results if r["status"] == "SKIP")
    err = sum(1 for r in results if r["status"] == "ERROR")
    ov = sum(1 for r in results if r["status"] == "OVERLAP")

    print("\n[SUMMARY]")
    print(f"  OK       : {ok}")
    print(f"  OVERLAP  : {ov}")
    print(f"  SKIP     : {skip}")
    print(f"  ERROR    : {err}")
    print(f"[OK] Wrote report: {report_path}")
    print("      Each target split is saved to: <target>\\split_811\\")


if __name__ == "__main__":

    main()
