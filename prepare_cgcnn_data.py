# coding: utf-8
import os
import json
import csv
import argparse
import pandas as pd
from pymatgen.core import Structure

# --- 1. ディレクトリの準備 ---
parser = argparse.ArgumentParser(description="Prepare CGCNN CIF and property data")
parser.add_argument(
    "--csv", default="data/df_wB.csv",
    help="入力CSV（既定: srv〇〇/data/df_wB.csv）",
)
args = parser.parse_args()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.expanduser(args.csv)
if not os.path.isabs(CSV_PATH):
    CSV_PATH = os.path.join(PROJECT_DIR, CSV_PATH)
CIF_DIR = os.path.join(PROJECT_DIR, "data", "cifs")
os.makedirs(CIF_DIR, exist_ok=True)

# 読み込むファイル（体積弾性率のあるデータを使う場合は df_wB.csv）
print(f"'{CSV_PATH}' を読み込み中...")
df = pd.read_csv(CSV_PATH)

# --- 2. 目的物性ごとの保存先ディレクトリと出力設定 ---
# 予測したい物性名と、保存先フォルダ名のマッピング
TARGET_PROPERTIES = {
    "K_VRH": "bulk_modulus",                       # 体積弾性率
    "formation_energy_per_atom": "formation_energy", # 生成エネルギー
    "band_gap": "band_gap"                          # バンドギャップ
}

# bulk modulus は 1～1000 GPa のデータだけを採用（両端を含む）
K_VRH_MIN_GPA = 1.0
K_VRH_MAX_GPA = 1000.0

# 物性ごとの id_prop リストを初期化
id_prop_dict = {prop: [] for prop in TARGET_PROPERTIES.keys()}

print("CIF ファイルの抽出と物性データの整理を開始します...")

# --- 3. ループ処理：CIF化と id_prop 用データの抽出 ---
for idx, row in df.iterrows():
    mat_id = str(row["material_id"])
    
    # (A) structure 列（JSON文字列）から Structure オブジェクトを復元し .cif を保存
    if pd.notna(row["structure"]):
        cif_filepath = os.path.join(CIF_DIR, f"{mat_id}.cif")
        
        # すでに CIF ファイルが存在しない場合のみ書き出し（処理の高速化）
        if not os.path.exists(cif_filepath):
            struct_dict = json.loads(row["structure"])
            struct = Structure.from_dict(struct_dict)
            struct.to(fmt="cif", filename=cif_filepath)

    # (B) 各物性値が存在すれば id_prop 用リストに追加
    for prop_key in TARGET_PROPERTIES.keys():
        val = row[prop_key]
        if pd.notna(val) and pd.notna(row["structure"]):
            if prop_key == "K_VRH":
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue

                if not K_VRH_MIN_GPA <= val <= K_VRH_MAX_GPA:
                    continue

            id_prop_dict[prop_key].append([mat_id, val])

# --- 4. 物性ごとの id_prop.csv の書き出し ---
for prop_key, folder_name in TARGET_PROPERTIES.items():
    output_dir = os.path.join(PROJECT_DIR, "data", "regression", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    id_prop_path = os.path.join(output_dir, "id_prop.csv")
    with open(id_prop_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(id_prop_dict[prop_key])
        
    print(f"[{folder_name}] id_prop.csv 作成完了 (件数: {len(id_prop_dict[prop_key])}) -> {id_prop_path}")

print("\n全データの変換・配置が完了しました！")
