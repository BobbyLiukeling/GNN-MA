# -*- coding: UTF-8 -*-
'''
@Author  ：Iris
@Date    ：2025-09-27 12:21
'''
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

atom_types = ['C', 'O', 'N', 'H', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'unknown']
bond_types = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

# 只保留1种编码方式
encode_folder_names = [
    # "no_edge_no_sidechain",
    "edge_no_sidechain"
]

def get_bond_type_onehot(bond):
    return [bond.GetBondType() == t for t in bond_types]

def encode_mol(mol, use_edge=True, use_sidechain=True):
    N = mol.GetNumAtoms()
    node_feats = []
    atom_to_scaffold = [0] * N
    scaffold_id = None

    # 侧链信息
    if use_sidechain:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_atoms = set([a.GetIdx() for a in scaffold.GetAtoms()])
        scaffold_id = hash(Chem.MolToSmiles(scaffold, isomericSmiles=False)) % 100000
        atom_to_scaffold = [1 if i in scaffold_atoms else 0 for i in range(N)]
    else:
        atom_to_scaffold = [0 for _ in range(N)]
        scaffold_id = -1

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol not in atom_types:
            symbol = "unknown"
        onehot = [symbol == t for t in atom_types]
        in_scaffold = atom_to_scaffold[i]
        scaffold_type = [scaffold_id] if use_sidechain else []
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        node_feat = onehot + [in_scaffold] + scaffold_type + [is_aromatic, is_in_ring]
        node_feats.append(node_feat)
    node_feats = np.array(node_feats, dtype=np.float32)
    adj = np.zeros((N, N), dtype=np.float32)
    edge_dim = len(bond_types) + 2
    edge_feats = np.zeros((N, N, edge_dim), dtype=np.float32) if use_edge else None

    # 边特征
    if use_edge:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1
            bond_onehot = get_bond_type_onehot(bond)
            is_aromatic = int(bond.GetIsAromatic())
            is_in_ring = int(bond.IsInRing())
            edge_attr = bond_onehot + [is_aromatic, is_in_ring]
            edge_feats[i, j, :] = edge_feats[j, i, :] = edge_attr
    else:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1
    return node_feats, adj, edge_feats

def encode_mol_twoways(mol):
    # 只用前两种方式
    return [
        # encode_mol(mol, use_edge=False, use_sidechain=False),  # no_edge_no_sidechain
        encode_mol(mol, use_edge=True, use_sidechain=False),   # edge_no_sidechain
    ]

def save_encoding(mol, save_paths, base_filename):
    encodings = encode_mol_twoways(mol)
    for idx, (node_feat, adj, edge_feat) in enumerate(encodings):
        npz_path = os.path.join(save_paths[idx], base_filename + ".npz")
        np.savez_compressed(npz_path,
                            node_feat=node_feat,
                            adj=adj,
                            edge_feat=edge_feat if edge_feat is not None else np.array([]))

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def process_smi_file(smi_path, save_dirs, prefix, is_decoy=False, max_decoy=10000):
    # 处理smi文件（actives.smi / inactives.smi）
    with open(smi_path, 'r') as f:
        for idx, line in enumerate(f):
            # decoys限制数量
            if is_decoy and idx >= max_decoy:
                break
            parts = line.strip().split()
            if not parts: continue
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else f"{prefix}_{idx}"
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            save_encoding(mol, save_dirs, name)

def process_target(target_dir, encode_base):
    target_name = os.path.basename(target_dir)
    encode_dirs = [os.path.join(encode_base, target_name, encode_folder_names[i]) for i in range(1)]  # 只前1种
    for ed in encode_dirs:
        ensure_dirs([os.path.join(ed, 'ligand'), os.path.join(ed, 'active'), os.path.join(ed, 'decoy')])

    # 1. ligand: 处理所有 _ligand.mol2 文件
    for fn in os.listdir(target_dir):
        if fn.endswith("_ligand.mol2"):
            fpath = os.path.join(target_dir, fn)
            mol = Chem.MolFromMol2File(fpath, sanitize=True)
            if mol is None: continue
            base_filename = os.path.splitext(fn)[0]
            save_dirs = [os.path.join(ed, 'ligand') for ed in encode_dirs]
            save_encoding(mol, save_dirs, base_filename)
    print("over ligand")

    # 2. actives.smi
    actives_path = os.path.join(target_dir, "actives.smi")
    if os.path.exists(actives_path):
        save_dirs = [os.path.join(ed, 'active') for ed in encode_dirs]
        process_smi_file(actives_path, save_dirs, prefix="active", is_decoy=False)
    print("over actives")

    # 3. inactives.smi
    inactives_path = os.path.join(target_dir, "inactives.smi")
    if os.path.exists(inactives_path):
        save_dirs = [os.path.join(ed, 'decoy') for ed in encode_dirs]
        process_smi_file(inactives_path, save_dirs, prefix="decoy", is_decoy=True, max_decoy=10000)
    print("over decoys")

def batch_encode_all(dealed_dir, encode_dir):
    for target_name in os.listdir(dealed_dir):
        tpath = os.path.join(dealed_dir, target_name)
        if os.path.isdir(tpath):
            print(f"处理靶点: {target_name}")
            process_target(tpath, encode_dir)

if __name__ == '__main__':
    dealed_dir = "LIT-PCBA"  # 原始分子根目录（每个靶标一个文件夹）
    encode_dir = "encode-LIT"  # 编码后保存目录
    batch_encode_all(dealed_dir, encode_dir)
    print("全部编码完成！")
