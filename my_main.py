import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

# 先ほど作成した自作モジュールから読み込む
from my_cgcnn import CGConvNet
from my_dataset import CIFDataset, collate_fn
from my_device import get_device



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
    parser.add_argument(
        '--device', choices=['auto', 'cpu', 'cuda'], default='auto',
        help='実行デバイス（autoはCUDAが使えればGPU、そうでなければCPU）',
    )
    parser.add_argument(
        '--target_transform', choices=['auto', 'none', 'ln'], default='auto',
        help='回帰ターゲットの変換。autoはbulk_modulusで自然対数を使用',
    )
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

    # 相対パスは、このスクリプトが置かれたフォルダ（srv〇〇）を基準にする
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)

    if args.task == 'regression':
        if args.target_transform == 'auto':
            target_transform = (
                'ln' if os.path.basename(os.path.normpath(data_dir)) == 'bulk_modulus'
                else 'none'
            )
        else:
            target_transform = args.target_transform
    else:
        target_transform = 'none'
    use_log_target = target_transform == 'ln'

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr

    # GPUが使える場合はGPU、なければCPUを使用
    device = get_device(args.device)
    print(
        f"使用デバイス: {device} | タスク: {args.task} | データパス: {data_dir}"
    )

    # ---------------------------------------------------------
    # 3. データセットと DataLoader の準備
    # ---------------------------------------------------------
    full_dataset = CIFDataset(data_dir=data_dir)
    total_len = len(full_dataset)

    # ln(K_VRH) を使う場合、0以下には自然対数を定義できないため除外する。
    valid_indices = list(range(total_len))
    if use_log_target:
        valid_indices = [
            i for i, row in enumerate(full_dataset.id_prop_data)
            if float(row[1]) > 0
        ]
        if not valid_indices:
            raise ValueError('自然対数変換できる正のターゲット値がありません。')
    eligible_dataset = Subset(full_dataset, valid_indices)
    effective_len = len(eligible_dataset)

    # データを 8:1:1（学習用:検証用:未使用テスト用）にランダム分割
    train_size = int(0.8 * effective_len)
    val_size = int(0.1 * effective_len)
    test_size = effective_len - train_size - val_size
    train_dataset, val_dataset, _test_dataset = random_split(
        eligible_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  
    )

    # 回帰ターゲットの平均・標準偏差は訓練データだけから計算する。
    # 検証・テストデータの情報が学習に混ざるのを防ぐ。
    target_mean = None
    target_std = None
    if args.task == 'regression':
        train_targets = torch.tensor(
            [
                float(full_dataset.id_prop_data[valid_indices[i]][1])
                for i in train_dataset.indices
            ],
            dtype=torch.float32,
        )
        if use_log_target:
            train_targets = torch.log(train_targets)
        target_mean = train_targets.mean().item()
        target_std = train_targets.std().item()
        if not torch.isfinite(torch.tensor(target_std)) or target_std == 0:
            raise ValueError('訓練データのターゲット標準偏差が0または不正です。')
        print(
            f"ターゲット変換: "
            f"{'自然対数 ln' if use_log_target else 'なし'} | "
            f"変換後の平均={target_mean:.6g}, "
            f"標準偏差={target_std:.6g}"
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
    test_loader = DataLoader(
        _test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        f"全データ数: {total_len} "
        f"(使用可能: {effective_len}, 対数対象外: {total_len - effective_len}; "
        f"学習用: {train_size}, 検証用: {val_size}, テスト用: {test_size})"
    )

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
                target_raw = target.to(device)
                target = torch.log(target_raw) if use_log_target else target_raw
                target = (target - target_mean) / target_std

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
        val_mae = 0.0

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
                    target = target.to(device).squeeze(1).long()        #分類の教師データは0,1でCrossEntropyに渡すので、整数型のlongに変換する必要がある。squeeze(1)で形状を(B,1)→(B,)に変換する。
                else:
                    target = target.to(device)
                    target_raw = target
                    target = torch.log(target) if use_log_target else target
                    target = (target - target_mean) / target_std

                out = model(atom_fea, u_fea, edge_idx, crystal_atom_idx)
                loss = criterion(out, target)
                val_loss += loss.item() * len(target)
                if args.task == 'regression':
                    pred_transformed = out * target_std + target_mean
                    pred_raw = (
                        torch.exp(pred_transformed)
                        if use_log_target else pred_transformed
                    )
                    val_mae += torch.sum(torch.abs(pred_raw - target_raw)).item()

        val_loss /= val_size
        if args.task == 'regression':
            val_mae /= val_size

        if args.task == 'regression':
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss (標準化MSE): {train_loss:.4f} | "
                f"Val Loss (標準化MSE): {val_loss:.4f} | "
                f"Val MAE (元の単位): {val_mae:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f}"
                f" | Val Loss: {val_loss:.4f}"
            )
        save_path = os.path.join(data_dir, 'my_best_model.pth')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'target_mean': target_mean,
                    'target_std': target_std,
                    'target_transform': target_transform,
                    'task': args.task,
                    'property_name': os.path.basename(os.path.normpath(data_dir)),
                },
                save_path,
            )

    print(
        "\n学習が完了しました！ベストモデルを my_best_model.pth に保存しました。"
    )

    # --- テストフェーズ：検証損失が最小だったモデルで未使用データを評価 ---
    save_path = os.path.join(data_dir, 'my_best_model.pth')
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']
    target_transform = checkpoint.get('target_transform', 'none')
    use_log_target = target_transform == 'ln'
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_correct = 0

    with torch.no_grad():
        for (
            atom_fea,
            u_fea,
            edge_idx,
            crystal_atom_idx,
            target,
        ) in test_loader:
            atom_fea = atom_fea.to(device)
            u_fea = u_fea.to(device)
            edge_idx = edge_idx.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

            if args.task == 'classification':
                target = target.to(device).squeeze(1).long()
            else:
                target = target.to(device)
                target_raw = target
                target = torch.log(target) if use_log_target else target
                target = (target - target_mean) / target_std

            out = model(atom_fea, u_fea, edge_idx, crystal_atom_idx)
            loss = criterion(out, target)
            test_loss += loss.item() * len(target)

            if args.task == 'regression':
                pred_transformed = out * target_std + target_mean
                pred_raw = (
                    torch.exp(pred_transformed)
                    if use_log_target else pred_transformed
                )
                test_mae += torch.sum(torch.abs(pred_raw - target_raw)).item()
            else:
                test_correct += (out.argmax(dim=1) == target).sum().item()          #outが(B,2)の形状で、argmax(dim=1)で各サンプルで大きい方が予測クラスを取得する。targetは(B,)の形状で、各サンプルの正解クラスを保持している。比較結果はTrue/Falseのテンソルになるので、sum()で正解数をカウントする。

    test_loss /= test_size
    if args.task == 'regression':
        test_mae /= test_size
        print(
            f"Test Loss (標準化MSE): {test_loss:.4f} | "
            f"Test MAE (元の単位): {test_mae:.4f}"
        )
    else:
        test_accuracy = test_correct / test_size
        print(
            f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4%}"
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
    model = CGConvNet(D_v, D_u, D_v, t_conv=1)

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
