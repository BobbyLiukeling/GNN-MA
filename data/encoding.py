# -*- coding: UTF-8 -*-
'''
@Author  ：Iris
@Date    ：2025-09-27 12:29
'''

# mol2 Molecules are converted into npz files, which have different data formats.
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

encode_folder_names = [
    # "no_edge_no_sidechain",
    "edge_no_sidechain",
    # "no_edge_sidechain",
    # "edge_sidechain"
]

def get_bond_type_onehot(bond):
    return [bond.GetBondType() == t for t in bond_types]

def encode_mol(mol, use_edge=True, use_sidechain=True):
    N = mol.GetNumAtoms()
    node_feats = []
    atom_to_scaffold = [0] * N
    scaffold_id = None

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

def encode_mol_fourways(mol):
    # 2. Add edges without adding side chains.
    node2, adj2, edge2 = encode_mol(mol, use_edge=True, use_sidechain=False)
    return [
        (node2, adj2, edge2),
    ]

def save_encoding(file_path, save_paths, base_filename):
    mol = Chem.MolFromMol2File(file_path, sanitize=True)
    if mol is None:
        print(f"Cannot be parsed: {file_path}")
        return False
    encodings = encode_mol_fourways(mol)
    for idx, (node_feat, adj, edge_feat) in enumerate(encodings):
        npz_path = os.path.join(save_paths[idx], base_filename + ".npz")
        np.savez_compressed(npz_path,
                            node_feat=node_feat,
                            adj=adj,
                            edge_feat=edge_feat if edge_feat is not None else np.array([]))
    return True

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def process_target(target_dir, encode_base):
    target_name = os.path.basename(target_dir)
    encode_dirs = [os.path.join(encode_base, target_name, encode_folder_names[i]) for i in range(1)]
    # Active and Decoy folders
    for ed in encode_dirs:
        ensure_dirs([os.path.join(ed, 'active'), os.path.join(ed, 'decoy')])
        os.makedirs(ed, exist_ok=True)  # The ligands exist directly under "ed"

    # active molecules
    actives_dir = os.path.join(target_dir, "active")
    for fn in os.listdir(actives_dir):
        if fn.endswith(".mol2"):
            fpath = os.path.join(actives_dir, fn)
            save_dirs = [os.path.join(ed, "active") for ed in encode_dirs]
            base_filename = os.path.splitext(fn)[0]
            save_encoding(fpath, save_dirs, base_filename)
    # decoy molecules
    decoy_dir = os.path.join(target_dir, "decoy")
    for fn in os.listdir(decoy_dir):
        if fn.endswith(".mol2"):
            fpath = os.path.join(decoy_dir, fn)
            save_dirs = [os.path.join(ed, "decoy") for ed in encode_dirs]
            base_filename = os.path.splitext(fn)[0]
            save_encoding(fpath, save_dirs, base_filename)
    # ligand molecules(there is only one file, which is not placed in a folder)
    ligand_file = os.path.join(target_dir, "crystal_ligand.mol2")
    if os.path.exists(ligand_file):
        save_dirs = [ed for ed in encode_dirs]
        base_filename = "crystal_ligand"
        save_encoding(ligand_file, save_dirs, base_filename)

def batch_encode_all(dealed_dir, encod_dir):
    for target_name in os.listdir(dealed_dir):
        tpath = os.path.join(dealed_dir, target_name)
        if os.path.isdir(tpath):
            print(f"\nProcessing target: {target_name}")
            process_target(tpath, encod_dir)

if __name__ == '__main__':
    dealed_dir = "DUD-E-dealed"  # Root directory of original molecules
    encod_dir = "encode-DUD-E"    # Directory to save after encoding
    batch_encode_all(dealed_dir, encod_dir)
    print("\nAll encodings completed!")
