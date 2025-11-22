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
                    X_train_2018_tensor: torch.Tensor, # 用於「偵測」 (D_new 100% 評估資料)
                    y_train_2018_tensor: torch.Tensor, # 用於「偵測」 (D_new 100% 評估資料)
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
        drift_detected = check_for_drift(X_train_2018_tensor.numpy(), artifacts_dir)
    except Exception as e:
        logging.error(f"Adaptive Module 檢查失敗: {e}", exc_info=True)
        return

    # if not drift_detected:
    #     logging.info("Adaptive Module 未偵測到顯著漂移。A-NIDS 響應中止。")
    #     return

    # --- (新) 步驟 3.2: 載入 D_new_real (2018) 並 *分割* ---
    logging.info("--- 3.2: 載入並清理 D_new (real) 2018 資料... ---")
    # 假設在上一步驟中，您已經完成了分割並產生了以下 Tensor：
    # D_new_train_tensor (特徵) 和 y_new_train_tensor (標籤)

    # 使用 y_new_train_tensor 的長度作為參考總數
    num_new_train_samples = len(y_train_2018_tensor)
    logging.info(f"D_new (train) 總樣本數: {num_new_train_samples} 筆。")


    # --- (新) 步驟 3.3: Generation Module 生成 D_old_fake *以匹配 D_new_train 的分佈* ---
    logging.info("--- 3.3: Generation Module 正在生成 D_old (fake)... ---")
    try:
        resampled_dfs = []
        
        # 載入 2017 的 LabelEncoder 以獲取舊標籤列表
        le_path = os.path.join(artifacts_dir, "label_encoder_2017.joblib")
        le_2017 = joblib.load(le_path)
        scaler_path = os.path.join(artifacts_dir, "minmax_scaler_2017.joblib")
        scaler_2017 = joblib.load(scaler_path)
        
        num_old_labels = len(le_2017.classes_)
        if num_old_labels == 0:
            logging.error("LabelEncoder 中沒有標籤。")
            return

        # 1. 💡 計算 D_new 訓練集 (y_train_2018_tensor) 中各類別的數量
        # 使用 torch.bincount 快速計算每個索引（即類別）的數量
        # 由於 y_train_2018_tensor 是 Long Tensor，可以直接使用
        # output: [Count_of_Label_0, Count_of_Label_1, ...]
        
        # 確保 y_train_2018_tensor 在 CPU 上進行計數
        y_new_train_cpu = y_train_2018_tensor.cpu() 
        
        # 獲取每個類別的計數
        label_counts_tensor = torch.bincount(y_new_train_cpu)
        
        # 將計數轉換為字典，以便於按標籤索引查找生成數量
        label_counts = label_counts_tensor.tolist()
        
        # 2. 💡 動態計算每個舊標籤要生成的數量 (匹配 D_new_train 的數量)
        total_fake_samples = 0
        
        logging.info(f"資料平衡策略：匹配 D_new (train) 訓練集 {num_new_train_samples} 筆的分佈。")

        # 呼叫生成函式，並根據索引獲取數量
        for idx, label_name in enumerate(le_2017.classes_):
            # 檢查該索引是否在計數列表中，如果不在 (表示 D_new_train 中沒有此標籤)，則生成數量為 0
            if idx < len(label_counts):
                samples_to_generate = label_counts[idx]
            else:
                samples_to_generate = 0
                
            logging.info(f"   -> 標籤 '{label_name}' (Index: {idx})，目標生成 {samples_to_generate} 筆。")

            if samples_to_generate > 0:
                df_old_fake_part = generate_stacked_data(
                    artifacts_dir=artifacts_dir,
                    label=label_name, # 這裡假設 generate_stacked_data 接受原始標籤名稱
                    num_samples_per_label=samples_to_generate
                )
                resampled_dfs.append(df_old_fake_part)
                total_fake_samples += len(df_old_fake_part)
                
        # 3. 堆疊和洗牌
        if resampled_dfs:
            df_old_fake = pd.concat(resampled_dfs, ignore_index=True)
            # 洗牌以打亂不同類別的假資料
            df_old_fake = df_old_fake.sample(frac=1, random_state=42).reset_index(drop=True)
            logging.info(f"總共成功生成 {total_fake_samples} 筆 D_old (fake) 資料。")
        else:
            df_old_fake = pd.DataFrame()
            logging.warning("沒有 D_new 訓練集中的標籤與 D_old 標籤匹配，未生成 D_old (fake) 資料。")
            if df_old_fake is None:
                logging.error("Generation Module 未能生成資料。中止。")
                return
        X_old_fake = df_old_fake.drop(columns=['label'])
        y_old_fake = df_old_fake['label']
        X_old_fake = df_old_fake[scaler_2017.feature_names_in_] # 確保欄位順序正確
        X_old_fake = scaler_2017.transform(X_old_fake)
        y_old_fake = le_2017.transform(y_old_fake)
        X_old_fake_tensor = torch.tensor(X_old_fake, dtype=torch.float32)
        y_old_fake_tensor = torch.tensor(y_old_fake, dtype=torch.long)
    except Exception as e:
        logging.error(f"Generation Module 生成失敗: {e}", exc_info=True)
        return

    # --- 3.4: 合併 (D_old_fake + D_new_train) 並處理 ---
    logging.info("--- 3.4: 合併 D_old(fake) 和 D_new(train) ... ---")
    X_retrain_tensor = torch.cat([X_old_fake_tensor, X_train_2018_tensor])
    y_retrain_tensor = torch.cat([y_old_fake_tensor, y_train_2018_tensor])
    logging.info(f"建立新的混合訓練集: {len(X_retrain_tensor)} 筆資料")
    

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
        X_test_2018_tensor, y_test_2018_tensor, # (新) 使用 D_new 的 30% 作為驗證集
        le_2017,
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # 儲存 A-NIDS 模型
    model_path = os.path.join(artifacts_dir, "noFCA-NIDS_model.pth")
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
        X_test_2018_tensor, # (新) 使用 D_new 的 30% 測試集
        y_test_2018_tensor, # (新)
        le_2017, # 仍使用 2017 的 encoder
        artifacts_dir,
        dataset_name="xss_noFCA" # (新) 檔名
    )
    