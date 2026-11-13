import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from my_cgcnn import CGConvNet
from my_dataset import CIFStructureDataset, collate_cif_fn
from my_device import get_device
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


PROPERTY_UNITS = {
    'bulk_modulus': 'GPa',
    'formation_energy': 'eV_per_atom',
    'band_gap': 'eV',
    'absolute_energy': 'eV_per_atom',
    'fermi_energy': 'eV',
    'shear_modulus': 'GPa',
    'poisson_ratio': '',
}


def read_candidate_ids(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f'候補ID CSVがありません: {csv_path}')
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []
    if 'material_id' in rows[0]:
        id_column = rows[0].index('material_id')
        rows = rows[1:]
    else:
        id_column = 0

    # 元CSVに同じIDが複数回あっても、CIFごとに一度だけ予測する。
    return list(dict.fromkeys(
        row[id_column].strip()
        for row in rows
        if len(row) > id_column and row[id_column].strip()
    ))


def predict_candidates():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='Predict one regression property for candidate crystal structures'
    )
    parser.add_argument(
        '--property', required=True, choices=PROPERTY_UNITS,
        help='予測する物性名（回帰物性）',
    )
    parser.add_argument(
        '--candidate_ids', default=None,
        help='候補ID CSV（省略時: data/candidate_ids/<property>.csv）',
    )
    parser.add_argument(
        '--cif_dir', default='data/cifs',
        help='IDに対応するCIFをまとめた共有フォルダ',
    )
    parser.add_argument(
        '--model', default=None,
        help='学習済みモデル（省略時: data/regression/<property>/my_best_model.pth）',
    )
    parser.add_argument(
        '--output', default=None,
        help='予測CSV（省略時: results/predictions/<property>.csv）',
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument(
        '--device', choices=['auto', 'cpu', 'cuda'], default='auto'
    )
    args = parser.parse_args()

    def resolve_path(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else project_dir / path

    candidate_ids_path = resolve_path(
        args.candidate_ids or f'data/candidate_ids/{args.property}.csv'
    )
    cif_dir = resolve_path(args.cif_dir)
    model_path = resolve_path(
        args.model or f'data/regression/{args.property}/my_best_model.pth'
    )
    output_path = resolve_path(
        args.output or f'results/predictions/{args.property}.csv'
    )
    material_ids = read_candidate_ids(candidate_ids_path)
    if not material_ids:
        raise ValueError(f'候補IDがありません: {candidate_ids_path}')
    if not model_path.is_file():
        raise FileNotFoundError(f'学習済みモデルがありません: {model_path}')

    device = get_device(args.device)
    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError(
            'この重みにはターゲットの標準化情報がありません。'
            '現在の my_main.py で学習し直してください。'
        )
    if checkpoint.get('task') != 'regression':
        raise ValueError('回帰タスクで学習したチェックポイントを指定してください。')
    checkpoint_property = checkpoint.get('property_name')
    if checkpoint_property and checkpoint_property != args.property:
        raise ValueError(
            f"指定物性は {args.property} ですが、モデルは "
            f"{checkpoint_property} 用です。"
        )

    target_mean = checkpoint.get('target_mean')
    target_std = checkpoint.get('target_std')
    target_transform = checkpoint.get('target_transform', 'none')
    if target_mean is None or target_std is None or target_std <= 0:
        raise ValueError('チェックポイントの標準化情報が不正です。')
    if target_transform not in ('ln', 'none'):
        raise ValueError(f'未対応のターゲット変換です: {target_transform}')

    dataset = CIFStructureDataset(
        cif_dir=cif_dir,
        material_ids=material_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_cif_fn,
    )
    model = CGConvNet(
        orig_D_v=92,
        D_u=41,
        D_v=64,
        t_conv=3,
        hidden_fea_len=128,
        out_dim=1,
        version='v5',
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    results = []
    with torch.no_grad():
        for atom_fea, u_fea, edge_idx, crystal_atom_idx, cif_ids in loader:
            atom_fea = atom_fea.to(device)
            u_fea = u_fea.to(device)
            edge_idx = edge_idx.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

            pred_standardized = model(
                atom_fea, u_fea, edge_idx, crystal_atom_idx
            ).reshape(-1)
            pred_transformed = pred_standardized * target_std + target_mean
            pred_values = (
                torch.exp(pred_transformed)
                if target_transform == 'ln' else pred_transformed
            )
            results.extend(
                (cif_id, float(value))
                for cif_id, value in zip(cif_ids, pred_values.cpu().tolist())
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    unit = PROPERTY_UNITS[args.property]
    prediction_column = f'predicted_{args.property}'
    if unit:
        prediction_column += f'_{unit}'
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['material_id', prediction_column])
        writer.writerows(results)

    print(f'予測物性: {args.property}')
    print(f'使用デバイス: {device}')
    print(f'候補数: {len(results)}')
    print(f'予測結果: {output_path}')


if __name__ == '__main__':
    predict_candidates()
