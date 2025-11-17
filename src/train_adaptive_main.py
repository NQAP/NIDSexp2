# train_adaptive_main.py
#
# 這是一個「一次性」的執行腳本。
# 它的目的是：
# 1. 載入 2017 (D_old) 的「完整」資料。
# 2. 清理資料 (使用 preprocessing.py 中的函式)。
# 3. 載入 2017 年的「MinMaxScaler」。
# 4. 「歸一化」完整的 2017 資料。
# 5. 呼叫 adaptive_module.py 中的 train_adaptive_model 來訓練並儲存 K-Means。

import os
import argparse
import logging
import pandas as pd
import numpy as np
import joblib

# 導入我們現有的預處理功能 (僅用於清理)
from preprocessing import standardize_columns, clean_features
# 導入新的適應性模組
from adaptive_module import train_adaptive_model

from utils import set_seed

def setup_logging():
    """設定日誌紀錄器"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():
    setup_logging()
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="A-NIDS: 訓練 Adaptive Module (K-Means)")
    parser.add_argument('--data_2017', type=str, required=True, 
                        help="2017 (D_old) 資料集的 .csv 檔案路徑。")
    parser.add_argument('--artifacts_dir', type=str, default="./result",
                        help="讀取 scaler 並儲存 K-Means 模型的目錄。")
    
    # 論文中的 K 和 α [cite: 324]
    parser.add_argument('--k_clusters', type=int, default=50,
                        help="K-Means 的叢集數量 K。論文建議 40-50 [cite: 500]。")
    parser.add_argument('--alpha', type=float, default=7.0,
                        help="計算 CI 的信賴係數 α。論文建議 7.0 [cite: 505]。")
    parser.add_argument('--or_threshold', type=float, default=2.0,
                        help="觸發漂移的離群值比率 (OR) 閾值 T (範例值)。")
                        
    args = parser.parse_args()

    logging.info("--- A-NIDS：開始訓練 Adaptive Module (K-Means) ---")
    
    # 1. 載入 2017 (D_old) 資料
    logging.info(f"從 {args.data_2017} 載入資料...")
    try:
        # 嘗試使用 'utf-8' 讀取，如果失敗，嘗試 'latin1'
        try:
            df = pd.read_csv(args.data_2017, encoding='utf-8')
        except UnicodeDecodeError:
            logging.warning("UTF-8 解碼失敗，嘗試使用 'latin1' 編碼...")
            df = pd.read_csv(args.data_2017, encoding='latin1')
            
    except Exception as e:
        logging.error(f"讀取 CSV 失敗: {e}", exc_info=True)
        return

    # 2. 清理資料 (與 CTGAN 訓練時相同)
    logging.info("標準化欄位名稱...")
    df_clean = standardize_columns(df)
    
    logging.info("清理特徵 (NaN, Inf, 全 0 欄位)...")
    df_clean = clean_features(df_clean)
    
    if 'label' not in df_clean.columns:
        logging.error("資料中找不到 'label' 欄位。請檢查 column_map.py。")
        return
        
    logging.info("資料清理完成。")

    # 3. 載入 2017 年的 MinMaxScaler
    scaler_path = os.path.join(args.artifacts_dir, "minmax_scaler_2017.joblib")
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"成功載入 MinMaxScaler: {scaler_path}")
    except FileNotFoundError:
        logging.error(f"找不到 MinMaxScaler ({scaler_path})。")
        logging.error("請先執行 main.py (FCN 訓練) 來生成 scaler。")
        return

    # 4. 準備 K-Means 的訓練資料 (歸一化)
    try:
        # 從 scaler 獲取 2017 年的標準特徵列表
        feature_names_2017 = scaler.feature_names_in_
        
        # 檢查 2017 資料是否包含所有必要的特徵
        missing_cols = set(feature_names_2017) - set(df_clean.columns)
        if missing_cols:
            logging.error(f"2017 資料中缺少 Scaler 所需的特徵: {missing_cols}")
            return
            
        # 獲取特徵 (DataFrame)
        X_features = df_clean[feature_names_2017]
        
        # (關鍵) 歸一化 D_old 的「完整」資料
        logging.info(f"正在歸一化 {len(X_features)} 筆 D_old 資料...")
        X_scaled = scaler.transform(X_features)
        
    except Exception as e:
        logging.error(f"資料歸一化失敗: {e}", exc_info=True)
        return

    # 5. 執行 Adaptive Module 訓練
    train_adaptive_model(
        X_scaled_old=X_scaled,
        k_clusters=args.k_clusters,
        alpha=args.alpha,
        or_threshold=args.or_threshold,
        output_dir=args.artifacts_dir
    )

if __name__ == "__main__":
    main()