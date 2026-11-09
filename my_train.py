import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# 先ほど作成した自作モジュールから読み込む
from my_cgcnn import CGConvNet
from my_dataset import CIFDataset, collate_fn
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


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
        #criterion = nn.L1Loss()  # 損失関数: MAE
        criterion = nn.MSELoss()  # 損失関数: MSE
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
    val_size = int(0.1*total_len)
    test_size = total_len - train_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  
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

            out = model(atom_fea, u_fea, edge_idx, crystal_atom_idx)    
            loss = criterion(out, target)
            B = out.shape[0]
            d_out = 2.0 * (out.detach() - target) / B

            # 手動逆伝播を実行して勾配辞書を取得
            manual_grads = model.manual_backward(d_out)

            # 重みの更新 (θ_new = θ_old - lr * grad)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in manual_grads:
                        # 勾配爆発を防ぐため、勾配の値を [-1.0, 1.0] の範囲に制限する
                        grad_clipped = torch.clamp(manual_grads[name], -1.0, 1.0)
                        param.data -= learning_rate * grad_clipped

            # 勾配辞書のクリア
            model.grads.clear()
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
        save_path = os.path.join(data_dir, 'my_best_model.pth')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(
        "\n学習が完了しました！ベストモデルを my_best_model.pth に保存しました。"
    )

def test_gradient():
    """手動逆伝播 (manual_backward) と PyTorch Autograd の一致検証関数"""
    print("=== 勾配チェック（Gradient Check）を開始します ===")

    # 1. 検証用のダミーデータを作成（または DataLoader から 1 バッチ取得）
    # 例: 3つの原子、4つのエッジ、2つの結晶
    D_v, D_u = 64, 10
    N_total, E_total = 3, 4

    v_fea = torch.randn(N_total, D_v, requires_grad=True)
    u_fea = torch.randn(E_total, D_u)
    edge_idx = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    B_idx = [torch.tensor([0, 1]), torch.tensor([2])]  # 2つの結晶
    target = torch.randn(2, 1)

    # 2. モデルと損失関数の準備
    model = CGConvNet(D_v, D_u, D_v, tr =1)

    criterion = nn.MSELoss()

    # 3. 順伝播
    out = model(v_fea, u_fea, edge_idx, B_idx)
    loss = criterion(out, target)

    # 4. A. PyTorch 標準の自動微分 (Autograd)
    model.zero_grad()
    loss.backward()

    # 5. B. 自作した手動逆伝播 (d_out = 2 * (out - target) / B)
    # MSE Loss (L = 1/B * sum((out - target)^2)) の微分
    d_out = 2.0 * (out.detach() - target) / out.shape[0]
    manual_grads = model.manual_backward(d_out)

    # 6. C. 両者の勾配の差分を比較
    print("\n--- パラメータごとの勾配差分 ---")
    all_passed = True
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        auto_grad = param.grad
        manual_grad = manual_grads[name]

        max_diff = torch.max(torch.abs(auto_grad - manual_grad)).item()
        print(f"{name:30s} | 最大差分: {max_diff:.8e}")

        if max_diff > 1e-4:
            all_passed = False

    if all_passed:
        print("\n✅ すべてのパラメータで勾配が一致しました！")
    else:
        print("\n❌ 勾配にズレがあるパラメータが存在します。確認が必要です。")


if __name__ == '__main__':
    # 勾配の検証だけを行いたい場合:
    #test_gradient()

    # 通常の学習を行いたい場合は上をコメントアウトして以下を実行:
     train_model()