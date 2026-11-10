import argparse
import os
import torch
from torch.utils.data import DataLoader
from my_cgcnn import CGConvNet
from my_dataset import CIFDataset, collate_fn
from my_device import get_device

def predict():
    parser = argparse.ArgumentParser(description='CGCNN prediction')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--data_dir', default='data/sample-regression')
    parser.add_argument('--model', default='my_best_model.pth')
    args = parser.parse_args()
    device = get_device(args.device)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.model):
        args.model = os.path.join(project_dir, args.model)

    # 1. 予測対象データセットの準備（ここでは sample-regression を使用）
    data_dir = args.data_dir
    dataset = CIFDataset(data_dir=data_dir)
    test_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # 2. モデル構造の定義と重みのロード
    model = CGConvNet(
        orig_D_v=92,
        D_u=41,
        D_v=64,
        t_conv=3,
        hidden_fea_len=128,
        out_dim=1,
        version='v5',
    ).to(device)

    # 保存したベストモデルのパラメータを読み込み
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print("=== CGCNN 物性予測（推論）開始 ===\n")

    # 3. 評価モード（勾配不要）で予測を実行
    with torch.no_grad():
        for i, (
            atom_fea,
            u_fea,
            edge_idx,
            crystal_atom_idx,
            target,
        ) in enumerate(test_loader):
            atom_fea = atom_fea.to(device)
            u_fea = u_fea.to(device)
            edge_idx = edge_idx.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

            # 順伝播（手動計算した重みを用いた予測）
            pred = model(atom_fea, u_fea, edge_idx, crystal_atom_idx)

            # 結果の抽出
            if hasattr(dataset, 'cif_ids'):
                crys_id = dataset.cif_ids[i]
            elif hasattr(dataset, 'id_prop_data'):
                crys_id = dataset.id_prop_data[i][0]
            else:
                crys_id = f"sample_{i+1:02d}"
            pred_val = pred.item()
            true_val = target.item()
            error = abs(pred_val - true_val)

            print(
                f"[{i+1:02d}] CIF ID: {crys_id:<10} | 予測物性値: {pred_val:8.4f} | 実測値: {true_val:8.4f} | 誤差: {error:8.4f}"
            )

    print("\n推論処理が完了しました！")


if __name__ == '__main__':
    predict()
