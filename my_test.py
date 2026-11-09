import argparse
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# 自作モジュールからの読み込み
from my_cgcnn import CGConvNet
from my_dataset import CIFDataset, collate_fn
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def train_model():
    # ---------------------------------------------------------
    # 1. コマンドライン引数の設定
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description='CGCNN Training')
    parser.add_argument('--task', default='regression', choices=['regression', 'classification'], help='タスクの種類')
    parser.add_argument('data_dir', nargs='?', default='data/regression/formation_energy_per_atom', help='データセットのパス')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. タスク設定
    # ---------------------------------------------------------
    if args.task == 'regression':
        data_dir = args.data_dir
        out_dim = 1
        criterion = nn.MSELoss()
    else:
        data_dir = args.data_dir
        out_dim = 2
        criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device} | タスク: {args.task} | データパス: {data_dir}")

    # ---------------------------------------------------------
    # 3. データセットの準備（Train / Val / Test に分割）
    # ---------------------------------------------------------
    full_dataset = CIFDataset(data_dir=data_dir)
    total_len = len(full_dataset)

    # 7:1.5:1.5 の割合で分割 (※Testデータは my_test.py で同じ分割を再現して使用)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size

    train_dataset, val_dataset, _ = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定シード
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"全データ数: {total_len} (学習: {train_size}, 検証: {val_size}, 未使用Test用: {test_size})")

    # ---------------------------------------------------------
    # 4. モデルの初期化
    # ---------------------------------------------------------
    sample_atom_fea, sample_u_fea, _, _, _ = full_dataset[0]
    orig_D_v = sample_atom_fea.shape[1]
    D_u = sample_u_fea.shape[1]

    model = CGConvNet(
        orig_D_v=orig_D_v,
        D_u=D_u,
        D_v=64,
        t_conv=3,
        hidden_fea_len=128,
        out_dim=out_dim,
        version='v5',
    ).to(device)

    # ---------------------------------------------------------
    # 5. 学習・検証ループ
    # ---------------------------------------------------------
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # --- 学習フェーズ ---
        model.train()
        train_loss = 0.0

        for atom_fea, u_fea, edge_idx, crystal_atom_idx, target in train_loader:
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
            
            # 手動逆伝播とパラメーター更新
            B = out.shape[0]
            d_out = 2.0 * (out.detach() - target) / B
            manual_grads = model.manual_backward(d_out)

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in manual_grads:
                        grad_clipped = torch.clamp(manual_grads[name], -1.0, 1.0)
                        param.data -= args.lr * grad_clipped

            model.grads.clear()
            train_loss += loss.item() * len(target)

        train_loss /= train_size

        # --- 検証フェーズ ---
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for atom_fea, u_fea, edge_idx, crystal_atom_idx, target in val_loader:
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
                if args.task == 'regression':
                    val_mae += torch.sum(torch.abs(out - target)).item()

        val_loss /= val_size
        val_mae /= val_size

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

        # ベストモデルの更新・保存
        save_path = os.path.join(data_dir, 'my_best_model.pth')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"\n学習完了！ ベストVal Loss: {best_val_loss:.4f}")

    # 最優モデルを指定物性フォルダ内へ移動
    target_model_path = os.path.join(data_dir, 'my_best_model.pth')
    shutil.move('my_best_model.pth', target_model_path)
    print(f"最適モデルを以下に保存しました:\n - {target_model_path}")


if __name__ == '__main__':
    train_model()