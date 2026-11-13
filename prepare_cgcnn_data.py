# coding: utf-8
import os
import json
import csv
import argparse
import pandas as pd
from pymatgen.core import Structure

# --- 1. ディレクトリの準備 --csvのパス
parser = argparse.ArgumentParser(description="Prepare CGCNN CIF and property data")
parser.add_argument(
    "--csv", default="~/machine-learning.org/codes/pymatgen/df-all.csv",
    help="入力CSV（既定: srv〇〇/data/df_wB.csv）",
)
parser.add_argument(
    "--candidate-dir", default="data/candidate_ids",
    help="物性ごとの候補ID CSVの保存先",
)
parser.add_argument(
    "--candidates-only", action="store_true",
    help="候補IDとCIFだけを作成し、既存の id_prop.csv は変更しない",
)
args = parser.parse_args()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.expanduser(args.csv)
if not os.path.isabs(CSV_PATH):
    CSV_PATH = os.path.join(PROJECT_DIR, CSV_PATH)
CIF_DIR = os.path.join(PROJECT_DIR, "data", "cifs")
os.makedirs(CIF_DIR, exist_ok=True)
CANDIDATE_DIR = os.path.expanduser(args.candidate_dir)
if not os.path.isabs(CANDIDATE_DIR):
    CANDIDATE_DIR = os.path.join(PROJECT_DIR, CANDIDATE_DIR)
os.makedirs(CANDIDATE_DIR, exist_ok=True)

# 読み込むファイル（体積弾性率のあるデータを使う場合は df_wB.csv）
print(f"'{CSV_PATH}' を読み込み中...")
df = pd.read_csv(CSV_PATH)

# --- 2. 目的物性ごとの保存先ディレクトリと出力設定 ---
# 予測したい物性名と、保存先フォルダ名のマッピング
TARGET_PROPERTIES = {
    "K_VRH": "bulk_modulus",                       # 体積弾性率
    "formation_energy_per_atom": "formation_energy", # 生成エネルギー
    "band_gap": "band_gap",                          # バンドギャップ
    "energy_per_atom": "absolute_energy",              # 原子あたりのエネルギー
    "efermi": "fermi_energy",                            # フェルミエネルギー
    "G_VRH": "shear_modulus",                            # せん断弾性率
    "poisson_ratio": "poisson_ratio",                      # ポアソン比
}

# bulk modulus は自然対数を計算できる 0 より大きく、1000 GPa 以下を採用
K_VRH_MIN_GPA = 0
K_VRH_MAX_GPA = 1000.0

# 物性ごとの id_prop と、構造はあるが値が欠損している候補IDを初期化
id_prop_dict = {prop: [] for prop in TARGET_PROPERTIES.keys()}
candidate_id_dict = {prop: set() for prop in TARGET_PROPERTIES.keys()}

print("CIF ファイルの抽出と物性データの整理を開始します...")

# --- 3. ループ処理：CIF化と id_prop 用データの抽出 ---
for idx, row in df.iterrows():
    if pd.isna(row["material_id"]):
        continue
    mat_id = str(row["material_id"])
    structure_data = row.get("structure")
    has_structure = structure_data is not None and pd.notna(structure_data)
    cif_filepath = os.path.join(CIF_DIR, f"{mat_id}.cif")
    if not has_structure and os.path.isfile(cif_filepath):
        has_structure = True
    
    # (A) structure 列（JSON文字列）から Structure オブジェクトを復元し .cif を保存
    if has_structure:
        # すでに CIF ファイルが存在しない場合のみ書き出し（処理の高速化）
        if not os.path.exists(cif_filepath):
            struct_dict = json.loads(structure_data)
            struct = Structure.from_dict(struct_dict)
            struct.to(fmt="cif", filename=cif_filepath)

    # (B) 各物性値が存在すれば id_prop 用リストに追加
    for prop_key in TARGET_PROPERTIES.keys():
        if prop_key not in df.columns:
            if has_structure:
                candidate_id_dict[prop_key].add(mat_id)
            continue
        val = row[prop_key]
        if not has_structure:
            continue
        if pd.isna(val):
            candidate_id_dict[prop_key].add(mat_id)
            continue

        if prop_key == "K_VRH":
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue

            # 値が存在するが範囲外の材料は「未知候補」と混同しない。
            if not K_VRH_MIN_GPA < val <= K_VRH_MAX_GPA:
                continue

        id_prop_dict[prop_key].append([mat_id, val])

# --- 4. 物性ごとの候補IDを書き出す ---
for prop_key, folder_name in TARGET_PROPERTIES.items():
    candidate_path = os.path.join(CANDIDATE_DIR, f"{folder_name}.csv")
    with open(candidate_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["material_id"])
        writer.writerows([[mat_id] for mat_id in sorted(candidate_id_dict[prop_key])])
    print(
        f"[{folder_name}] 候補ID作成完了 "
        f"(件数: {len(candidate_id_dict[prop_key])}) -> {candidate_path}"
    )

# candidates-only なら学習用の既存 id_prop.csv を上書きしない。
if args.candidates_only:
    print("候補IDとCIFのみを作成しました。既存の id_prop.csv は変更していません。")
    raise SystemExit(0)

# --- 5. 物性ごとの id_prop.csv の書き出し ---
for prop_key, folder_name in TARGET_PROPERTIES.items():
    output_dir = os.path.join(PROJECT_DIR, "data", "regression", folder_name)       #保存先パス
    os.makedirs(output_dir, exist_ok=True)
    
    id_prop_path = os.path.join(output_dir, "id_prop.csv")
    with open(id_prop_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(id_prop_dict[prop_key])
        
    print(f"[{folder_name}] id_prop.csv 作成完了 (件数: {len(id_prop_dict[prop_key])}) -> {id_prop_path}")

print("\n全データの変換・配置が完了しました！")
