# train_ctgan_main.py
#
# 這是一個「一次性」的執行腳本。
# 它的唯一目的是載入 2017 (D_old) 資料，並訓練 Stacked-CTGAN 生成模型。
# 訓練完成後，模型會被儲存在 anids_artifacts/ 目錄中。

import os
import argparse
import logging
import pandas as pd
import numpy as np
import torch

# 導入我們現有的預處理功能 (僅用於清理)
from preprocessing import standardize_columns, clean_features
# 導入新的生成模組
from generation_module import train_stacked_ctgan

from features_whitelist import FEATURES_WHITELIST, DISCRETE_FEATURES, CONTINUOUS_FEATURES

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
    
    parser = argparse.ArgumentParser(description="A-NIDS: 訓練 Stacked-CTGAN 生成模組")
    parser.add_argument('--data_2017', type=str, default='./rawdata/2017.csv', 
                        help="2017 (D_old) 資料集的 .csv 檔案路徑。")
    parser.add_argument('--output_dir', type=str, default="./result",
                        help="儲存 ctgan_*.pkl 模型的輸出目錄。")
    parser.add_argument('--epochs', type=int, default=500,
                        help="每個 CTGAN 模型的訓練 Epoch 數 (CTGAN 需要較多時間)。")
    args = parser.parse_args()

    logging.info("--- A-NIDS：開始訓練 Generation Module (Stacked-CTGAN) ---")
    
    # 1. 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 載入 2017 (D_old) 資料
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

    # 3. 清理資料
    #    注意：我們使用「未歸一化」的原始資料來訓練 CTGAN，
    #    這樣它才能學習到真實的分佈。
    logging.info("標準化欄位名稱...")
    df_clean = standardize_columns(df)
    
    logging.info("清理特徵 (NaN, Inf, 全 0 欄位)...")
    df_clean = clean_features(df_clean)
    
    feature_cols = DISCRETE_FEATURES + CONTINUOUS_FEATURES
    
    if len(feature_cols) == 0:
        logging.error("沒有找到任何特徵欄位。請檢查 column_map.py 是否正確。")
        return
        
    logging.info(f"將在 {len(feature_cols)} 個特徵上訓練 CTGAN。")
    logging.debug(f"特徵列表: {feature_cols[:5]}...") # 僅顯示前 5 個

    # 5. 執行 Stacked-CTGAN 訓練
    #    (這一步會花費很長時間)
    train_stacked_ctgan(
        df_2017=df_clean,
        feature_cols=feature_cols,
        output_dir=args.output_dir,
        ctgan_epochs=args.epochs,
    )
    
    logging.info("--- Generation Module 訓練完成 ---")

if __name__ == "__main__":
    main()