# -*- coding: UTF-8 -*-
'''
@Author  ：Iris
@Date    ：2025-09-27 11:29 
'''


# 将activate 和decoys进行拆分
import os
import gzip
import shutil

def split_mol2_gz(gz_file, out_dir, prefix):
    with gzip.open(gz_file, 'rt', encoding='utf-8') as gz_in:
        mol2_content = gz_in.read()
    blocks = mol2_content.split("@<TRIPOS>MOLECULE")
    count = 0
    for block in blocks[1:]:
        count += 1
        mol2_text = "@<TRIPOS>MOLECULE" + block
        out_path = os.path.join(out_dir, f"{prefix}_{count}.mol2")
        with open(out_path, "w", encoding='utf-8') as f_out:
            f_out.write(mol2_text)
    print(f"{os.path.basename(gz_file)} 提取{count}个分子到 {out_dir}")

def process_target_dir(target_dir, output_base):
    target_name = os.path.basename(target_dir)
    out_dir = os.path.join(output_base, target_name)
    os.makedirs(out_dir, exist_ok=True)
    # 1. 提取actives
    actives_gz = os.path.join(target_dir, "actives_final.mol2.gz")
    if os.path.exists(actives_gz):
        active_dir = os.path.join(out_dir, "active")
        os.makedirs(active_dir, exist_ok=True)
        split_mol2_gz(actives_gz, active_dir, "active")
    # 2. 提取decoys
    decoys_gz = os.path.join(target_dir, "decoys_final.mol2.gz")
    if os.path.exists(decoys_gz):
        decoy_dir = os.path.join(out_dir, "decoy")
        os.makedirs(decoy_dir, exist_ok=True)
        split_mol2_gz(decoys_gz, decoy_dir, "decoy")
    # 3. 复制crystal_ligand.mol2
    crystal_file = os.path.join(target_dir, "crystal_ligand.mol2")
    if os.path.exists(crystal_file):
        shutil.copy(crystal_file, os.path.join(out_dir, "crystal_ligand.mol2"))
        print(f"复制{crystal_file}到{out_dir}")

def batch_process_all_targets(all_dir, dealed_dir):
    index = 0
    for name in os.listdir(all_dir):
        target_path = os.path.join(all_dir, name)
        if os.path.isdir(target_path):
            print(f"\n处理靶点: {name}")
            process_target_dir(target_path, dealed_dir)
        index += 1


if __name__ == '__main__':
    all_dir = "DUD-E"          # 原始数据根目录
    dealed_dir = "DUD-E-dealed"    # 输出数据根目录
    os.makedirs(dealed_dir, exist_ok=True)
    batch_process_all_targets(all_dir, dealed_dir)
