# (新) A-NIDS 重新訓練階段的輔助函式
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

def phase_3_retrain(artifacts_dir: str, 
                    data_2018_path: str,
                    X_test_2017_tensor: torch.Tensor, # 用於最終評估
                    y_test_2017_tensor: torch.Tensor, # 用於最終評估
                    X_test_2018_tensor: torch.Tensor, # 用於最終評估
                    y_test_2018_tensor: torch.Tensor, # 用於最終評估
                    input_features: int, 
                    num_classes: int,
                    args):
    """
    執行 A-NIDS 的 Phase 3：偵測、生成、重新訓練、評估
    """
    logging.info("="*50)
    logging.info("PHASE 3: A-NIDS 響應 (偵測、生成、重新訓練)")
    logging.info("="*50)

    # --- 3.1: Adaptive Module 偵測漂移 ---
    # 我們使用 Phase 2 已經處理好的 X_test_2018_tensor 來進行偵測
    logging.info("--- 3.1: Adaptive Module 正在檢查資料漂移... ---")
    try:
        drift_detected = check_for_drift(X_test_2018_tensor.numpy(), artifacts_dir)
    except Exception as e:
        logging.error(f"Adaptive Module 檢查失敗: {e}", exc_info=True)
        return

    if not drift_detected:
        logging.info("Adaptive Module 未偵測到顯著漂移。A-NIDS 響應中止。")
        return

    # --- 3.2: Generation Module 生成 D_old_fake ---
    logging.info("--- 3.2: Generation Module 正在生成 D_old (fake)... ---")
    try:
        # 載入 2017 的 LabelEncoder 以獲取標籤列表
        le_path = os.path.join(artifacts_dir, "label_encoder_2017.joblib")
        le_2017 = joblib.load(le_path)
        
        # 呼叫生成函式 (假設每個標籤生成 1000 筆)
        df_old_fake = generate_stacked_data(
            artifacts_dir=artifacts_dir,
            label_encoder=le_2017,
            num_samples_per_label=args.gen_samples_per_label 
        )
        if df_old_fake is None:
            logging.error("Generation Module 未能生成資料。中止。")
            return
    except Exception as e:
        logging.error(f"Generation Module 生成失敗: {e}", exc_info=True)
        return

    # --- 3.3: 載入 D_new_real (2018) ---
    logging.info("--- 3.3: 載入並清理 D_new (real) 2018 資料... ---")
    try:
        df_new_real, y_new_real = load_and_clean_data(data_2018_path, artifacts_dir)
        if df_new_real is None:
            logging.error("載入 2018 真實資料失敗。中止。")
            return
        
        # 將標籤欄位加回去，以便合併
        df_new_real['label'] = y_new_real
        
    except Exception as e:
        logging.error(f"載入 D_new (real) 失敗: {e}", exc_info=True)
        return

    # --- 3.4: 合併 (D_old_fake + D_new_real) 並處理 ---
    logging.info("--- 3.4: 合併 D_old(fake) 和 D_new(real) ... ---")
    df_retrain = pd.concat([df_old_fake, df_new_real], ignore_index=True)
    logging.info(f"建立新的混合訓練集: {len(df_retrain)} 筆資料")

    # 載入 2017 的 Scaler 和 Encoder
    scaler_path = os.path.join(artifacts_dir, "minmax_scaler_2017.joblib")
    scaler_2017 = joblib.load(scaler_path)
    
    # 分離特徵和標籤
    y_retrain = df_retrain['label']
    X_retrain = df_retrain[scaler_2017.feature_names_in_] # 確保欄位順序正確
    
    # 處理資料 (歸一化, 編碼)
    logging.info("歸一化並編碼混合訓練集...")
    X_retrain_scaled = scaler_2017.transform(X_retrain)
    y_retrain_encoded = le_2017.transform(y_retrain)
    
    # 轉換為 Tensors
    X_retrain_tensor = torch.tensor(X_retrain_scaled, dtype=torch.float32)
    y_retrain_tensor = torch.tensor(y_retrain_encoded, dtype=torch.long)

    # --- 3.5: 重新訓練 (A-NIDS_model) ---
    logging.info("--- 3.5: 訓練新的 A-NIDS 模型... ---")
    # 建立一個全新的 FCN 模型
    anids_model = detect_module(
        input_features=input_features,
        num_classes=num_classes
    )
    
    # 使用與 Mlp-2017 相同的參數進行訓練
    anids_model, anids_history = train_model(
        anids_model,
        X_retrain_tensor, y_retrain_tensor,
        X_test_2018_tensor, y_test_2018_tensor, # 使用 2018 資料作為驗證集
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # 儲存 A-NIDS 模型
    model_path = os.path.join(artifacts_dir, "A-NIDS_model.pth")
    torch.save(anids_model.state_dict(), model_path)
    logging.info(f"A-NIDS (更新後) 模型已儲存至: {model_path}")
    
    # 繪製 A-NIDS 訓練圖
    plot_training_history(anids_history, artifacts_dir, plot_filename="anids_training_history.png")

    # --- 3.6: 最終評估 A-NIDS_model ---
    logging.info("="*50)
    logging.info("PHASE 3: A-NIDS 最終評估")
    logging.info("="*50)
    
    # 評估 1: 檢查對「新資料」的適應性 (我們期望高分)
    logging.info("--- 評估 A-NIDS 模型在 2018 資料 (D_new) 上的效能 ---")
    evaluate_model(
        anids_model,
        X_test_2018_tensor,
        y_test_2018_tensor,
        le_2017,
        artifacts_dir,
        dataset_name="A-NIDS_on_2018_D_new"
    )
    
    # 評估 2: 檢查對「舊資料」的記憶 (我們期望分數高於 Mlp-2018)
    logging.info("--- 評估 A-NIDS 模型在 2017 資料 (D_old) 上的效能 (檢查災難性遺忘) ---")
    evaluate_model(
        anids_model,
        X_test_2017_tensor,
        y_test_2017_tensor,
        le_2017,
        artifacts_dir,
        dataset_name="A-NIDS_on_2017_D_old"
    )