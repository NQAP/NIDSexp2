# main.py
#
# A-NIDS 論文復現 - 主執行腳本
# 
# 執行流程：
# Phase 1: (Mlp-2017) 訓練初始 FCN 模型 (model_2017)
# Phase 2: (Data Drift) 評估 model_2017 在 2018 資料上的效能 (展示資料漂移)
# Phase 3: (A-NIDS Response)
#    3.1: Adaptive Module 偵測漂移
#    3.2: Generation Module 生成 D_old_fake
#    3.3: 載入 D_new_real (2018)
#    3.4: 合併 (D_old_fake + D_new_real) 並重新訓練 (A-NIDS_model)
#    3.5: 評估 A-NIDS_model 在 2017 (防遺忘) 和 2018 (適應性) 上的效能

import argparse
import logging
import os
import torch
import joblib
import pandas as pd

# 導入 A-NIDS 模組
from preprocessing import old_data_preprocessing, preprocess_new_data, load_and_clean_data
from detect_module import detect_module
from train_and_eval import train_model, evaluate_model, plot_training_history
from adaptive_module import check_for_drift
from generation_module import generate_stacked_data
from retrain_anids import phase_3_retrain


def main(args):
    """主要執行函式"""
    logging.info("A-NIDS 論文復現腳本 (完整流程) 啟動...")
    
    # 1. 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"所有產出 (artifacts) 將儲存至: {args.output_dir}")

    # ==================================================================
    # PHASE 1: 訓練初始模型 (Mlp-2017)
    # ==================================================================
    logging.info("="*50)
    logging.info("PHASE 1: 訓練初始偵測模型 (Mlp-2017)")
    logging.info("="*50)
    
    try:
        data_2017 = old_data_preprocessing(
            args.data_2017, 
            args.output_dir
        )
        if data_2017[0] is None: raise Exception("old_data_preprocessing 失敗")
        
        X_train_2017, y_train_2017, X_test_2017, y_test_2017, \
        feature_count, class_count, le_2017 = data_2017
        
        logging.info(f"資料預處理完成。特徵數: {feature_count}, 類別數: {class_count}")
        
    except Exception as e:
        logging.error(f"Phase 1 資料預處理失敗: {e}", exc_info=True)
        return

    # 建立偵測模型 (Mlp-2017)
    model_2017 = detect_module(
        input_features=feature_count,
        num_classes=class_count
    )

    # 訓練 Mlp-2017
    try:
        model_2017, history_2017 = train_model(
            model_2017,
            X_train_2017, y_train_2017,
            X_test_2017, y_test_2017, # 使用 2017 測試集作為驗證集
            args.epochs,
            args.batch_size,
            args.learning_rate
        )
    except Exception as e:
        logging.error(f"Mlp-2017 模型訓練失敗: {e}", exc_info=True)
        return

    # 儲存 Mlp-2017
    model_path = os.path.join(args.output_dir, "model_2017.pth")
    torch.save(model_2017.state_dict(), model_path)
    logging.info(f"初始模型 (Mlp-2017) 已儲存至: {model_path}")
        
    # 評估 Mlp-2017 在 D_old 上的效能
    logging.info("--- 評估 Mlp-2017 在 2017 測試集 (D_old) 上的效能 ---")
    evaluate_model(model_2017, X_test_2017, y_test_2017, le_2017, 
                   args.output_dir, dataset_name="Mlp-2017_on_2017_D_old")
    
    # 繪製 Mlp-2017 訓練歷史
    plot_training_history(history_2017, args.output_dir, plot_filename="mlp_2017_training_history.png")
    
    logging.info("--- PHASE 1 完成 ---")

    # ==================================================================
    # PHASE 2: 評估資料漂移 (Data Drift)
    # ==================================================================
    if not args.data_2018:
        logging.info("未提供 --data_2018 參數。腳本在 Phase 1 後結束。")
        return

    logging.info("="*50)
    logging.info("PHASE 2: 評估資料漂移 (Data Drift)")
    logging.info("="*50)
    
    try:
        # 載入並處理 2018 資料 (僅用於評估)
        data_2018 = preprocess_new_data(
            args.data_2018,
            args.output_dir
        )
        if data_2018[0] is None: raise Exception("preprocess_new_data 失敗")
        
        X_test_2018, y_test_2018, le_2018 = data_2018
        
        logging.info("使用 Mlp-2017 模型評估 2018 年資料 (D_new)...")
        
        # (關鍵) 使用 2017 的模型去評估 2018 的資料
        evaluate_model(
            model_2017, # <-- 2017 的模型
            X_test_2018, # <-- 2018 的資料
            y_test_2018, # <-- 2018 的標籤
            le_2018, 
            args.output_dir,
            dataset_name="Mlp-2017_on_2018_D_new" # 預期效能會很差
        )
        logging.info("--- PHASE 2 完成 (已展示資料漂移) ---")

    except Exception as e:
        logging.error(f"Phase 2 評估失敗: {e}", exc_info=True)
        return

    # ==================================================================
    # PHASE 3: A-NIDS 響應
    # ==================================================================
    if args.skip_retrain:
        logging.info("已設定 --skip_retrain，A-NIDS 響應 (Phase 3) 將被跳過。")
        return
        
    try:
        phase_3_retrain(
            artifacts_dir=args.output_dir,
            data_2018_path=args.data_2018,
            X_test_2017_tensor=X_test_2017,
            y_test_2017_tensor=y_test_2017,
            X_test_2018_tensor=X_test_2018,
            y_test_2018_tensor=y_test_2018,
            input_features=feature_count,
            num_classes=class_count,
            args=args
        )
        logging.info("--- PHASE 3 完成 ---")
    except Exception as e:
        logging.error(f"Phase 3 A-NIDS 響應失敗: {e}", exc_info=True)
        
    logging.info("A-NIDS 完整流程結束。")


if __name__ == '__main__':
    # 1. 設定 Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 2. 設定 argparse
    parser = argparse.ArgumentParser(description="A-NIDS 論文 - 完整復現腳本")
    
    # --- 主要參數 ---
    parser.add_argument('--data_2017', 
                        type=str, 
                        required=True, 
                        help="[必要] CICIDS-2017 (舊資料) .csv 檔案的路徑")
    
    parser.add_argument('--data_2018', 
                        type=str, 
                        default=None, 
                        help="[可選] CICIDS-2018 (新資料) .csv 檔案的路徑。 (用於 Phase 2 和 3)")
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default="./result", 
                        help="[可選] 儲存所有 artifacts (scalers, models, reports) 的目錄")
    
    # --- FCN 訓練參數 ---
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help="[可選] FCN 訓練的 Epoch 數量")
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256, # 增加 batch size 加快訓練
                        help="[可選] FCN 訓練的 Batch Size")

    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.001, 
                        help="[可選] FCN 優化器的學習率")

    # --- A-NIDS Phase 3 參數 ---
    parser.add_argument('--skip_retrain', 
                        action='store_true', 
                        help="[可選] 如果設定此項，將跳過 Phase 3 (A-NIDS 重新訓練)")
                        
    parser.add_argument('--gen_samples_per_label', 
                        type=int, 
                        default=1000, 
                        help="[可選] 在 Phase 3 中，為每個舊標籤生成的樣本數")

    # 3. 解析參數並執行 main
    args = parser.parse_args()
    main(args)