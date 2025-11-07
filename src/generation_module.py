# generation_module.py
#
# 封裝 A-NIDS 生成模組 (Stacked-CTGAN) 的所有邏輯。
# 根據論文 V-D 節，我們為 2017 年資料中的「每一個標籤」訓練一個獨立的 CTGAN。

import os
import joblib
import logging
import pandas as pd
import torch
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder # 為了類型提示

def train_stacked_ctgan(df_2017: pd.DataFrame, 
                        feature_cols: list, 
                        output_dir: str, 
                        ctgan_epochs: int = 500,):
    """
    訓練 Stacked-CTGAN 模型。
    
    為 df_2017 中 'label' 欄位的每一個唯一值，訓練一個獨立的 CTGAN 模型，
    並將模型儲存到 output_dir。

    Args:
        df_2017 (pd.DataFrame): 包含「特徵」和 'label' 欄位的 2017 (D_old) 資料框。
                                **注意：此處的特徵應為「未歸一化」的原始資料**。
        feature_cols (list): 要用於訓練 CTGAN 的特徵欄位名稱列表。
        output_dir (str): 儲存 ctgan_*.pkl 模型的目錄。
        ctgan_epochs (int): 每個 CTGAN 模型的訓練 epoch 數。
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logging.info("偵測到 CUDA (GPU)，將使用 GPU 進行訓練。")
    else:
        logging.info("未偵測到 CUDA (GPU)，將使用 CPU 進行訓練 (可能會很慢)。")
    logging.info(f"--- 開始訓練 Stacked-CTGAN (共 {len(feature_cols)} 個特徵) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    unique_labels = df_2017['label'].unique()
    logging.info(f"在 D_old 中找到 {len(unique_labels)} 個唯一標籤，將為每個標籤訓練一個模型。")
    
    # 找出 CTGAN 需要的離散欄位
    # 在我們的案例中，所有特徵都是連續的 (float/int)，CTGAN 會自動處理。
    # 如果您有 'protocol' 等分類特徵，應將其加入 discrete_columns。
    # 為了簡單起見，我們假設所有 feature_cols 都是連續的。
    discrete_columns = [] 
    
    for i, label in enumerate(unique_labels):
        logging.info(f"--- (模型 {i+1}/{len(unique_labels)}) 開始訓練: {label} ---")
        
        # 1. 過濾出該標籤的資料
        label_data = df_2017[df_2017['label'] == label]
        
        # 2. 我們只需要特徵欄位來訓練模型
        training_data = label_data[feature_cols]
        
        if len(training_data) == 0:
            logging.warning(f"標籤 '{label}' 沒有資料，跳過。")
            continue
        elif len(training_data) < 10:
             logging.warning(f"標籤 '{label}' 只有 {len(training_data)} 筆資料，模型品質可能不佳。")

        # 3. 初始化並訓練 CTGAN
        # (CTGAN 訓練可能需要一些時間)
        model_path = os.path.join(output_dir, f"ctgan_{label}.pkl")
        
        # 節省時間：如果模型已存在，則跳過
        if os.path.exists(model_path):
            logging.info(f"模型 {model_path} 已存在，跳過訓練。")
            continue
            
        logging.info(f"初始化 CTGAN (Epochs={ctgan_epochs})...")
        try:
            ctgan_model = CTGAN(epochs=ctgan_epochs, verbose=True, cuda=use_cuda)
            
            logging.info(f"開始擬合 (fit) {len(training_data)} 筆資料...")
            ctgan_model.fit(training_data, discrete_columns)
            
            # 4. 儲存訓練好的模型
            joblib.dump(ctgan_model, model_path)
            logging.info(f"模型已儲存至: {model_path}")
            
        except Exception as e:
            logging.error(f"訓練標籤 {label} 時發生錯誤: {e}", exc_info=True)
            
    logging.info("--- Stacked-CTGAN 所有模型訓練完成 ---")


def generate_stacked_data(artifacts_dir: str, 
                        label_encoder: LabelEncoder, 
                        num_samples_per_label: int = 1000):
    """
    使用儲存的 Stacked-CTGAN 模型生成合成的 (D_old) 資料。

    Args:
        artifacts_dir (str): 儲存 ctgan_*.pkl 模型的目錄。
        label_encoder (LabelEncoder): 已擬合的 2017 年標籤編碼器 (用於獲取所有標籤列表)。
        num_samples_per_label (int): 要為每個標籤生成的樣本數。

    Returns:
        pd.DataFrame: 一個包含生成特徵和 'label' 欄位的 DataFrame。
                      如果沒有模型，則返回 None。
    """
    
    logging.info("--- 開始從 Stacked-CTGAN 生成合成資料 ---")
    all_generated_data = []
    known_labels = label_encoder.classes_
    
    logging.info(f"將為 {len(known_labels)} 個標籤 (每個 {num_samples_per_label} 筆) 生成資料...")
    
    for label in known_labels:
        model_path = os.path.join(artifacts_dir, f"ctgan_{label}.pkl")
        
        if not os.path.exists(model_path):
            logging.warning(f"找不到標籤 '{label}' 的 CTGAN 模型 ({model_path})，跳過。")
            continue
            
        try:
            # 1. 載入模型
            model = joblib.load(model_path)
            
            # 2. 生成樣本
            generated_data = model.sample(num_samples_per_label)
            
            # 3. 附加回正確的標籤
            generated_data['label'] = label
            
            all_generated_data.append(generated_data)
            logging.debug(f"已生成 '{label}' 的資料。")
            
        except Exception as e:
            logging.error(f"生成標籤 '{label}' 的資料時發生錯誤: {e}", exc_info=True)

    if not all_generated_data:
        logging.error("未能生成任何資料。請檢查 anids_artifacts 資料夾中是否有 ctgan_*.pkl 模型。")
        return None

    # 將所有生成的資料合併為一個 DataFrame
    df_generated = pd.concat(all_generated_data, ignore_index=True)
    
    logging.info(f"--- 總共生成 {len(df_generated)} 筆合成 (D_old) 資料 ---")
    return df_generated