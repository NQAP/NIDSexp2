# retrain_anids.py
#
# 封裝 A-NIDS 的 Phase 3 (響應) 邏輯。
# (新) 更新：現在會動態平衡資料：
# 1. 載入 D_new (100%)
# 2. 分割為 D_new_train (70%) 和 D_new_test (30%)
# 3. 生成 D_old_fake，使其總數等於 D_new_train (70%) 的總數
# 4. 在 (D_old_fake + D_new_train) 上訓練
# 5. 在 D_new_test 和 D_old_test 上評估

import logging
import os
import joblib
import pandas as pd
import torch
from argparse import Namespace # 用於類型提示
import math

# 導入 A-NIDS 模組
from preprocessing import load_and_clean_data
from adaptive_module import check_for_drift
from generation_module import generate_stacked_data
from detect_module import detect_module
from train_and_eval import train_model, evaluate_model, plot_training_history

# (新) 導入 train_test_split
from sklearn.model_selection import train_test_split

from utils import set_seed


def phase_3_retrain(artifacts_dir: str, 
                    data_2018_path: str,
                    X_test_2017_tensor: torch.Tensor, # 用於最終評估 (D_old 30% 測試集)
                    y_test_2017_tensor: torch.Tensor, # 用於最終評估 (D_old 30% 測試集)
                    X_test_2018_tensor: torch.Tensor, # 用於「偵測」 (D_new 100% 評估資料)
                    y_test_2018_tensor: torch.Tensor, # 用於「偵測」 (D_new 100% 評估資料)
                    input_features: int, 
                    num_classes: int,
                    args: Namespace): # 傳入 argparse 的參數
    set_seed(42)
    """
    執行 A-NIDS 的 Phase 3：偵測、生成、重新訓練、評估
    """
    logging.info("="*50)
    logging.info("PHASE 3: A-NIDS 響應 (偵測、生成、重新訓練)")
    logging.info("="*50)

    # --- 3.1: Adaptive Module 偵測漂移 ---
    # 我們使用 Phase 2 已經處理好的 X_test_2018_tensor (代表 D_new 的 100%) 來進行偵測
    logging.info("--- 3.1: Adaptive Module 正在檢查資料漂移... ---")
    try:
        # X_test_2018_tensor 是 PyTorch Tensor, .numpy() 轉換為 NumPy
        drift_detected = check_for_drift(X_test_2018_tensor.numpy(), artifacts_dir)
    except Exception as e:
        logging.error(f"Adaptive Module 檢查失敗: {e}", exc_info=True)
        return

    if not drift_detected:
        logging.info("Adaptive Module 未偵測到顯著漂移。A-NIDS 響應中止。")
        return

    # --- (新) 步驟 3.2: 載入 D_new_real (2018) 並 *分割* ---
    logging.info("--- 3.2: 載入並清理 D_new (real) 2018 資料... ---")
    try:
        # 使用 preprocessing 中的 load_and_clean_data 函式
        df_new_real, y_new_real = load_and_clean_data(data_2018_path, artifacts_dir)
        if df_new_real is None:
            logging.error("載入 2018 真實資料失敗。中止。")
            return
        
        # 將標籤欄位加回去，以便進行分割
        df_new_real_with_labels = df_new_real.copy()
        df_new_real_with_labels['label'] = y_new_real
        
        # (*** 關鍵 ***) 將 2018 資料分割為 70% 訓練集和 30% 測試集
        logging.info("將 D_new (real) 分割為 70% 訓練集和 30% 測試集...")
        D_new_train_df, D_new_test_df = train_test_split(
            df_new_real_with_labels, 
            test_size=0.3, 
            random_state=42, 
            stratify=df_new_real_with_labels['label']
        )
        
        num_new_train_samples = len(D_new_train_df)
        logging.info(f"D_new (real) 訓練集 (70%): {num_new_train_samples} 筆")
        logging.info(f"D_new (real) 測試集 (30%): {len(D_new_test_df)} 筆")

        if num_new_train_samples == 0:
            logging.error("2018 真實訓練資料為 0 筆。中止。")
            return
            
    except Exception as e:
        logging.error(f"載入 D_new (real) 失敗: {e}", exc_info=True)
        return

    # --- (新) 步驟 3.3: Generation Module 生成 D_old_fake *以匹配 70% 的數量* ---
    logging.info("--- 3.3: Generation Module 正在生成 D_old (fake)... ---")
    try:
        resampled_dfs = []
        # 載入 2017 的 LabelEncoder 以獲取標籤列表
        le_path = os.path.join(artifacts_dir, "label_encoder_2017.joblib")
        le_2017 = joblib.load(le_path)
        
        num_old_labels = len(le_2017.classes_)
        if num_old_labels == 0:
            logging.error("LabelEncoder 中沒有標籤。")
            return

        # (*** 關鍵 ***) 動態計算每個標籤要生成的數量
        # 確保 D_old_fake 總數約等於 D_new_train (70%) 總數
        dynamic_samples_per_label = math.ceil(num_new_train_samples / num_old_labels)
        
        logging.info(f"資料平衡策略： D_new (train) = {num_new_train_samples} 筆。")
        logging.info(f"將為 {num_old_labels} 個舊標籤，每個生成 {dynamic_samples_per_label} 筆資料。")
        logging.info(f"(預計生成 {dynamic_samples_per_label * num_old_labels} 筆 D_old (fake) 資料)")

        # 呼叫生成函式
        for label in le_2017.classes_:
            df_old_fake = generate_stacked_data(
                artifacts_dir=artifacts_dir,
                label=label,
                num_samples_per_label=dynamic_samples_per_label
            )
            resampled_dfs.append(df_old_fake)
        df_old_fake = pd.concat(resampled_dfs, ignore_index=True)
        df_old_fake = df_old_fake.sample(frac=1, random_state=42).reset_index(drop=True)
        if df_old_fake is None:
            logging.error("Generation Module 未能生成資料。中止。")
            return
    except Exception as e:
        logging.error(f"Generation Module 生成失敗: {e}", exc_info=True)
        return

    # --- 3.4: 合併 (D_old_fake + D_new_train) 並處理 ---
    logging.info("--- 3.4: 合併 D_old(fake) 和 D_new(train) ... ---")
    df_retrain = pd.concat([df_old_fake, D_new_train_df], ignore_index=True)
    logging.info(f"建立新的混合訓練集: {len(df_retrain)} 筆資料")
    logging.debug(f"混合標籤分佈:\n{df_retrain['label'].value_counts()}")

    # 載入 2017 的 Scaler 和 Encoder
    scaler_path = os.path.join(artifacts_dir, "minmax_scaler_2017.joblib")
    scaler_2017 = joblib.load(scaler_path)
    
    # --- 處理 (D_old_fake + D_new_train) ---
    logging.info("歸一化並編碼「混合訓練集」...")
    y_retrain = df_retrain['label']
    X_retrain = df_retrain[scaler_2017.feature_names_in_] # 確保欄位順序正確
    X_retrain_scaled = scaler_2017.transform(X_retrain)
    y_retrain_encoded = le_2017.transform(y_retrain)
    X_retrain_tensor = torch.tensor(X_retrain_scaled, dtype=torch.float32)
    y_retrain_tensor = torch.tensor(y_retrain_encoded, dtype=torch.long)

    # --- (新) 處理 D_new_test (30% holdout) ---
    logging.info("歸一化並編碼「新的 30% 測試集」...")
    y_new_test = D_new_test_df['label']
    X_new_test = D_new_test_df[scaler_2017.feature_names_in_]
    X_new_test_scaled = scaler_2017.transform(X_new_test)
    y_new_test_encoded = le_2017.transform(y_new_test)
    X_test_new_tensor = torch.tensor(X_new_test_scaled, dtype=torch.float32)
    y_test_new_tensor = torch.tensor(y_new_test_encoded, dtype=torch.long)
    

    # --- 3.5: 重新訓練 (A-NIDS_model) ---
    logging.info("--- 3.5: 訓練新的 A-NIDS 模型... ---")
    # # 建立一個全新的 FCN 模型
    anids_model = detect_module(
        input_features=input_features,
        num_classes=num_classes
    )
    
    # 使用與 Mlp-2017 相同的參數進行訓練
    anids_model, anids_history = train_model(
        anids_model,
        X_retrain_tensor, y_retrain_tensor,
        X_test_new_tensor, y_test_new_tensor, # (新) 使用 D_new 的 30% 作為驗證集
        le_2017,
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # 儲存 A-NIDS 模型
    model_path = os.path.join(artifacts_dir, "A-NIDS_model.pth")
    torch.save(anids_model.state_dict(), model_path)
    logging.info(f"A-NIDS (更新後) 模型已儲存至: {model_path}")
    anids_model.load_state_dict(torch.load(model_path))
    # 繪製 A-NIDS 訓練圖
    plot_training_history(anids_history, artifacts_dir, plot_filename="anids_training_history.png")

    # --- 3.6: 最終評估 A-NIDS_model ---
    logging.info("="*50)
    logging.info("PHASE 3: A-NIDS 最終評估")
    logging.info("="*50)
    
    # 評估 1: 檢查對「新資料」的適應性 (我們期望高分)
    logging.info("--- 評估 A-NIDS 模型在 D_new 30% 測試集上的效能 ---")
    evaluate_model(
        anids_model,
        X_test_new_tensor, # (新) 使用 D_new 的 30% 測試集
        y_test_new_tensor, # (新)
        le_2017, # 仍使用 2017 的 encoder
        artifacts_dir,
        dataset_name="A-NIDS_on_2018_D_new_testset" # (新) 檔名
    )
    
    # 評估 2: 檢查對「舊資料」的記憶 (我們期望分數高於 Mlp-2018)
    logging.info("--- 評估 A-NIDS 模型在 D_old 30% 測試集上的效能 (檢查災難性遺忘) ---")
    evaluate_model(
        anids_model,
        X_test_2017_tensor, # (不變) 仍使用 D_old 的 30% 測試集
        y_test_2017_tensor, # (不變)
        le_2017, # 仍使用 2017 的 encoder
        artifacts_dir,
        dataset_name="A-NIDS_on_2017_D_old_testset" # (新) 檔名
    )

    X_hybrid = torch.cat([X_test_2017_tensor, X_test_new_tensor], 0)
    y_hybrid = torch.cat([y_test_2017_tensor, y_test_new_tensor], 0)

    logging.info("--- 評估 A-NIDS 模型在 Hybrid 30% 測試集上的效能 (檢查整體表現) ---")
    evaluate_model(
        anids_model,
        X_hybrid, 
        y_hybrid,
        le_2017, # 仍使用 2017 的 encoder
        artifacts_dir,
        dataset_name="A-NIDS_on_Hybrid_testset" # (新) 檔名
    )