# main.py
#
# A-NIDS 論文復現 - 主執行腳本
# 
# (新) Phase 1 訓練 Mlp-2017 時，傳入 `use_focal_loss=True`。
# (新) Phase 3 訓練 A-NIDS 時，不傳入 (使用預設 False)。
# (新) `old_data_preprocessing` 解包 7 個項目。

import argparse
import logging
import os
import torch
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 導入 A-NIDS 模組
from preprocessing import old_data_preprocessing, preprocess_new_data
from detect_module import detect_module
from train_and_eval import train_model, evaluate_model, plot_training_history

# 導入 Phase 3 響應模組
from retrain_anids import phase_3_retrain

from utils import set_seed

def main(args):
    set_seed(42)
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
        
        # (*** 新 ***) 解包 (現在是 7 個回傳值)
        # (*** 新 ***) 解包 class_weights (現在有 8 個回傳值)
        X_train_2017, y_train_2017, X_test_2017, y_test_2017, \
        feature_count, class_count, le_2017, class_weights = data_2017
        
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
        # (*** 新 ***) 傳入 class_weights
        model_2017, history_2017 = train_model(
            model_2017,
            X_train_2017, y_train_2017,
            X_test_2017, y_test_2017, # 使用 2017 測試集作為驗證集
            label_encoder=le_2017,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            class_weights=class_weights # <-- 在此傳入權重
            # (已移除 gamma 和 use_focal_loss)
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
        
        logging.info(f"使用 Mlp-2017 模型評估 2018 年資料 (D_new)...")
        
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
        # 呼叫 retrain_anids 模組
        phase_3_retrain(
            artifacts_dir=args.output_dir,
            data_2018_path=args.data_2018,
            X_test_2017_tensor=X_test_2017,
            y_test_2017_tensor=y_test_2017,
            X_test_2018_tensor=X_test_2018, # 傳入 D_new 的 100% 資料
            y_test_2018_tensor=y_test_2018, 
            input_features=feature_count,
            num_classes=class_count,
            args=args # 傳入完整的 args
        )
        logging.info("--- PHASE 3 完成 ---")
    except Exception as e:
        logging.error(f"Phase 3 A-NIDS 響應失敗: {e}", exc_info=True)
        
    logging.info("A-NIDS 完整流程結束。")


if __name__ == '__main__':
    # 1. 設定 Logging
    

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
                        default=30, 
                        help="[可選] FCN 訓練的 Epoch 數量")
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256, 
                        help="[可選] FCN 訓練的 Batch Size")

    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.00005, 
                        help="[可選] FCN 優化器的學習率")

    # --- A-NIDS Phase 3 參數 ---
    parser.add_argument('--skip_retrain', 
                        action='store_true', 
                        help="[可選] 如果設定此項，將跳過 Phase 3 (A-NIDS 重新訓練)")
                        
    # 3. 解析參數並執行 main
    args = parser.parse_args()
    main(args)