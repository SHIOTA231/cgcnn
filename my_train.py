import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 先ほど作成した自作モジュールから読み込む
from my_cgcnn import CGConvNet
from my_dataset import CIFDataset, collate_fn


def train_model():
    # ---------------------------------------------------------
    # 1. コマンドライン引数の設定（実行時に外から渡せるようにする）
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description='CGCNN Training')
    parser.add_argument(
        '--task',
        default='regression',
        choices=['regression', 'classification'],
        help='タスクの種類',
    )
    parser.add_argument(
        '--data_dir', default=None, help='データセットのパス'
    )
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. タスク（回帰 or 分類）に応じた自動設定の切り替え
    # ---------------------------------------------------------
    if args.task == 'regression':
        data_dir = (
            args.data_dir if args.data_dir else 'data/sample-regression'
        )
        out_dim = 1  # 予測値は1つの連続数値
        criterion = nn.L1Loss()  # 損失関数: MAE
    else:  # classification
        data_dir = (
            args.data_dir if args.data_dir else 'data/sample-classification'
        )
        out_dim = 2  # 2クラス分類（例: 0 または 1）
        criterion = nn.CrossEntropyLoss()  # 損失関数: クロスエントロピー

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr

    # GPUが使える場合はGPU、なければCPUを使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"使用デバイス: {device} | タスク: {args.task} | データパス: {data_dir}"
    )

    # ---------------------------------------------------------
    # 3. データセットと DataLoader の準備
    # ---------------------------------------------------------
    full_dataset = CIFDataset(data_dir=data_dir)
    total_len = len(full_dataset)

    # データを 8:2（学習用:検証用）にランダム分割
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"全データ数: {total_len} (学習用: {train_size}, 検証用: {val_size})")

    # ---------------------------------------------------------
    # 4. モデル、オプティマイザの初期化
    # ---------------------------------------------------------
    model = CGConvNet(
        orig_D_v=92,  # atom_init.json の次元数
        D_u=41,  # ガウス展開後の距離次元数
        D_v=64,  # 隠れ層の原子特徴量次元
        t_conv=3,  # CGConv層の数
        hidden_fea_len=128,  # MLPの隠れ層次元
        out_dim=out_dim,  # 引数に応じて 1 または 2 に変化
        version='v5',
    ).to(device)

    # 最適化手法: 勾配降下手法Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    #学習率の自動調整や慣性を持たせることができるAdamを使用し、学習率はコマンドライン引数で指定できるようにする。model.parameters()は重み

    # ---------------------------------------------------------
    # 5. 学習ループ (Training & Validation Loop)
    # ---------------------------------------------------------
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):  #エポックループ 
        # --- 学習フェーズ ---
        model.train()
        train_loss = 0.0

        for (                           #バッチループ
            atom_fea,
            u_fea,
            edge_idx,
            crystal_atom_idx,
            target,
        ) in train_loader:              #to(device)でGPUに転送する
            atom_fea = atom_fea.to(device)
            u_fea = u_fea.to(device)
            edge_idx = edge_idx.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

            # 分類問題の場合は target を LongTensor(整数型) に変換し、形状を (B,) に調整
            if args.task == 'classification':
                target = target.to(device).squeeze(1).long()
            else:
                target = target.to(device)

            optimizer.zero_grad()       #前回計算した勾配メモをリセット
            out = model(atom_fea, u_fea, edge_idx, crystal_atom_idx)    
            loss = criterion(out, target)
            loss.backward()             #逆伝播法で計算
            optimizer.step()            #こうばいを更新

            train_loss += loss.item() * len(target)             #1サンプルの平均誤差×サンプル数で、バッチ内の全誤差を計算する。len(target)はバッチ内のサンプル数Bを表す。

        train_loss /= train_size

        # --- 検証フェーズ ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():           #評価用なので勾配の更新はしない。no_grad()でこのブロックの中では勾配計算や保存を行わないでください
            for (
                atom_fea,
                u_fea,
                edge_idx,
                crystal_atom_idx,
                target,
            ) in val_loader:
                atom_fea = atom_fea.to(device)
                u_fea = u_fea.to(device)
                edge_idx = edge_idx.to(device)
                crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

                if args.task == 'classification':
                    target = target.to(device).squeeze(1).long()
                else:
                    target = target.to(device)

                out = model(atom_fea, u_fea, edge_idx, crystal_atom_idx)
                loss = criterion(out, target)
                val_loss += loss.item() * len(target)

        val_loss /= val_size

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'my_best_model.pth')

    print(
        "\n学習が完了しました！ベストモデルを my_best_model.pth に保存しました。"
    )


if __name__ == '__main__':
    train_model()