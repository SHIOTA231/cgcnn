import os
import csv
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure


class GaussianDistance(object):
    """
    距離(Å)をガウス基底関数で多次元ベクトル(u_fea)に展開するクラス
    """
    def __init__(self, dmin=0, dmax=8.0, step=0.2,):
        self.r_ij = np.arange(dmin, dmax + step, step) 
        self.siguma = step
       
    def expand(self, distances):
        """
        distances: (E_total,) の1次元アレイ/テンソル
        return: (E_total, nbr_fea_len) の2次元ガウス展開テンソル
        """
        return np.exp(-((distances[:, np.newaxis] - self.r_ij) ** 2) / (self.siguma ** 2))  #u_(i,j)kをガウス動径で多次元ベクトルに変換することによって、連続値を表現している。
    


class CIFDataset(Dataset):
    """
    CIFファイル群と id_prop.csv からグラフ構造データを動的に生成するDataset
    """
    def __init__(self, data_dir, max_num_nbr=12, radius=8.0, dmin=0, step=0.2):
        self.data_dir = data_dir        #結晶構造CIFと教師データ格納フォルダ
        self.max_num_nbr = max_num_nbr  #最大近傍原子数
        self.radius = radius            #カットオフ半径

        # 1. atom_init.json の読み込み
        atom_init_path = os.path.join(self.data_dir, 'atom_init.json')
        assert os.path.exists(atom_init_path), f"{atom_init_path} が存在しません"
        with open(atom_init_path) as f:
            atom_init = json.load(f)        #json形式の辞書を読み込むためのおまじない
        # 原子番号(int) -> 92次元Tensor
        self.atom_features = {int(k): torch.Tensor(v) for k, v in atom_init.items()}    #kは原子番号を表す文字列"1"。それを数字に変換する。その原子番号の特徴量ベクトルをtorch.Tensorに変換して辞書に格納する。


        # 2. id_prop.csv の読み込み (ID と ターゲット物性値)
        id_prop_path = os.path.join(self.data_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_path), f"{id_prop_path} が存在しません"
        with open(id_prop_path) as f:       
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # 3. ガウス展開器の準備
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        
        # CIFファイルの読み込み (pymatgen)
        cif_path = os.path.join(self.data_dir, f"{cif_id}.cif")
        crystal = Structure.from_file(cif_path)

        # 1. 原子特徴量 (v_fea) の構築
        atom_fea = torch.vstack([self.atom_features[site.specie.Z] for site in crystal]) # Shape: (N, 92),Zは原子番号を表す。よって結晶内Nこの原子を取得し、N×92のテンソルを作成する。
        

        # 2. 近傍原子の検索 (8.0 Å以内)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        
        all_nbrs_sorted = []
        for nbrs in all_nbrs:
            # 距離が近い順にソートし、最大近傍数 max_num_nbr (12個) に制限
            nbrs = sorted(nbrs, key=lambda x: x[1])
            all_nbrs_sorted.append(nbrs[:self.max_num_nbr])

        # エッジリスト (edge_idx) と 距離アレイの作成
        nbr_i_idx, nbr_j_idx, distances = [], [], []
        for i, nbrs in enumerate(all_nbrs_sorted):
            for nbr in nbrs:
                nbr_i_idx.append(i)          # 受け手原子(i)
                nbr_j_idx.append(nbr[2])     # 送り手原子(j)
                distances.append(nbr[1])     # 距離 r_ij

        nbr_i_idx = torch.LongTensor(nbr_i_idx)
        nbr_j_idx = torch.LongTensor(nbr_j_idx)
        edge_idx = torch.stack([nbr_i_idx, nbr_j_idx], dim=0) # Shape: (2, E)

        # 3. 距離のガウス展開 (u_fea)
        distances = np.array(distances)
        u_fea = torch.Tensor(self.gdf.expand(distances)) # Shape: (E, 41)

        target = torch.Tensor([float(target)])

        return atom_fea, u_fea, edge_idx, target


def collate_fn(dataset_list):
    """
    DataLoader がバッチ化する際に呼び出す関数
    複数の結晶の atom_fea, u_fea, edge_idx を縦に連結し、B_idx (crystal_atom_idx) を生成する
    """
    batch_atom_fea = []
    batch_u_fea = []
    batch_edge_idx = []
    crystal_atom_idx = []
    batch_target = []

    base_atom_idx = 0
    for i, (atom_fea, u_fea, edge_idx, target) in enumerate(dataset_list):
        num_atoms = atom_fea.shape[0]

        batch_atom_fea.append(atom_fea)
        batch_u_fea.append(u_fea)
        
        # エッジインデックスを全原子IDのオフセット(base_atom_idx)分ずらして結合
        batch_edge_idx.append(edge_idx + base_atom_idx)
        
        # どの原子がバッチ内の何番目の結晶に属するかを保持
        crystal_atom_idx.append(torch.arange(base_atom_idx, base_atom_idx + num_atoms))
        
        batch_target.append(target)
        base_atom_idx += num_atoms

    # 1つの巨大なテンソルへ縦結合 (Concat)
    atom_fea = torch.cat(batch_atom_fea, dim=0)
    u_fea = torch.cat(batch_u_fea, dim=0)
    edge_idx = torch.cat(batch_edge_idx, dim=1)
    target = torch.stack(batch_target, dim=0)

    return atom_fea, u_fea, edge_idx, crystal_atom_idx, target


# ---------------------------------------------------------
# 動作確認テスト
# ---------------------------------------------------------
if __name__ == "__main__":
    # サンプルデータのパスを指定（sample-regression または sample-classification）
    data_path = 'data/sample-regression'
    
    if os.path.exists(data_path):
        dataset = CIFDataset(data_dir=data_path)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        print(f"データセット件数: {len(dataset)} 件")

        for atom_fea, u_fea, edge_idx, crystal_atom_idx, target in loader:
            print("\n--- DataLoaderの出力テンソルShape確認 ---")
            print(f"全原子特徴量 atom_fea : {atom_fea.shape}")  # (N_total, 92)
            print(f"全エッジ特徴量 u_fea    : {u_fea.shape}")     # (E_total, 41)
            print(f"エッジリスト edge_idx  : {edge_idx.shape}")  # (2, E_total)
            print(f"結晶数 B_idx 個数      : {len(crystal_atom_idx)}") # 2
            print(f"正解ターゲット target  : {target.shape}")   # (2, 1)
            print("---------------------------------------")
            print("Dataset / DataLoader の実装成功！")
            break
    else:
        print(f"パス {data_path} が見つかりません。")