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

        # data/regression/formation_energy_per_atom の2つ上の階層の cifs フォルダを参照
        base_dir = os.path.dirname(os.path.dirname(data_dir)) # 'data' フォルダ
        self.cif_dir = os.path.expanduser('~/cgcnn/data/cifs')

        # 1. atom_init.json の読み込み
        atom_init_path = '/home/falufito/cgcnn/data/atom_init.json'
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
        return len(self.id_prop_data)       #何個のリストがあるかを返す。つまり、何個の結晶構造があるかを返す。

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        
        # CIFファイルの読み込み (pymatgen)siteの情報(x,y,z)を取得
        cif_path = os.path.join(self.cif_dir, f"{cif_id}.cif")
        crystal = Structure.from_file(cif_path)             #pymatgenのStructureクラスを使って、CIFファイルから結晶構造を読み込む。crystalは結晶構造の情報が入ったオブジェクト。

        # 1. 原子特徴量 (v_fea) の構築
        atom_fea = torch.vstack([self.atom_features[site.specie.Z] for site in crystal]) # Shape: (N, 92),Zは原子番号を表す。よって結晶内Nこの原子を取得し、N×92のテンソルを作成する。
        

        # 2. 近傍原子の検索 (8.0 Å以内)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        
        all_nbrs_sorted = []    #最初は空っぽ
        for nbrs in all_nbrs:
            # 距離が近い順にソートし、最大近傍数 max_num_nbr (12個) に制限
            nbrs = sorted(nbrs, key=lambda x: x[1])     #x[0]は周りの原子の情報が丸ごと入ったデータ、x[1]は距離、x[2]はsiteのindex。x[1]の距離を基準にソートする。
            all_nbrs_sorted.append(nbrs[:self.max_num_nbr])     #からのリストに、新しくリストを作成し、その原子の近傍原子の情報を入れる。最大12個まで。それを繰り返し、すべての原子の近傍原子情報を入れる。


        # 近傍原子のall_nbrs_sortedから、my_cgcnn.pyへ直接入力できるedge_idxとu_feaのテンソルを作る処理
        nbr_i_idx, nbr_j_idx, distances = [], [], []
        for i, nbrs in enumerate(all_nbrs_sorted):  #iを選ぶループ
            for nbr in nbrs:                        #nbrsの中の近傍原子を選ぶループ
                nbr_i_idx.append(i)          # 受け手原子(i)
                nbr_j_idx.append(nbr[2])     # 送り手原子(j)nbr[2]
                distances.append(nbr[1])     # 距離 r_ijnbr[1]

        nbr_i_idx = torch.LongTensor(nbr_i_idx)     #LongTensorに変換することで、整数のテンソルに変換する。
        nbr_j_idx = torch.LongTensor(nbr_j_idx)
        edge_idx = torch.stack([nbr_i_idx, nbr_j_idx], dim=0) # Shape: (2, E),dim=0で縦に結合する。1行目が送り手、2行目が受け手の原子番号を表す。

        # 3. 距離のガウス展開 (u_fea)
        distances = np.array(distances) # Shape: (E,)、二原子間の実際の距離をただ順番に並べた１次元配列（例: [1.5, 2.1, 1.5, 2.1, ...]
        u_fea = torch.Tensor(self.gdf.expand(distances)) # Shape: (E, 41)、expand関数で距離をガウス展開して、E個の近傍原子間の距離を41次元のベクトルに変換する。*self.gdf.は引数ではない

        target = torch.Tensor([float(target)])  #Shape:(1,)、教師データの物性値をfloatに変換して、1次元テンソルにする。

        return (atom_fea, u_fea, edge_idx), torch.Tensor([float(target)])


def collate_fn(dataset_list):       #照合
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
    for i, ((atom_fea, u_fea, edge_idx), target) in enumerate(dataset_list):
        num_atoms = atom_fea.shape[0]   #shape[0]は原子数Nを表す。atom_feaは(N, 92)のテンソルなので、shape[0]でNを取得する。

        batch_atom_fea.append(atom_fea)
        batch_u_fea.append(u_fea)
        
        # エッジインデックスを全原子IDのオフセット(base_atom_idx)分ずらして結合
        batch_edge_idx.append(edge_idx + base_atom_idx)     #2個目以降の結晶のedge_idxは、1個目の結晶の原子数分だけずらす必要がある。base_atom_idxは、これまでに結合した原子数を保持している。
        
        # どの原子がバッチ内の何番目の結晶に属するかを保持
        crystal_atom_idx.append(torch.arange(base_atom_idx, base_atom_idx + num_atoms))     #2個目以降の結晶の原子番号は、1個目の結晶の原子数分だけずらす必要がある。base_atom_idxは、これまでに結合した原子数を保持している。
        
        batch_target.append(target)
        base_atom_idx += num_atoms

    # 1つの巨大なテンソルへ縦結合 (Concat)
    atom_fea = torch.cat(batch_atom_fea, dim=0)
    u_fea = torch.cat(batch_u_fea, dim=0)
    edge_idx = torch.cat(batch_edge_idx, dim=1)         #dim=1で横に結合することで、(2, E_total)の形状になる。E_totalはバッチ内の全エッジ数の合計。
    target = torch.stack(batch_target, dim=0)

    return atom_fea, u_fea, edge_idx, crystal_atom_idx, target


# ---------------------------------------------------------
# 動作確認テスト
# ---------------------------------------------------------
if __name__ == "__main__":
    #データのパスを指定
    data_path = 'data/regression/band_gap'  # 例: 'data/regression/band_gap' または 'data/classification/metal'
    
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