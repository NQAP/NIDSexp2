import pandas as pd
import numpy as np
import os
import joblib
import logging
import torch
import re # <-- (新) 導入正規表達式
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# --- 導入欄位名稱對應表 ---
from column_map import COLUMN_MAP
from features_whitelist import FEATURES_WHITELIST
from generation_module import generate_stacked_data

def standardize_columns(df):
    """
    清理欄位名稱：移除前後空格、轉為小寫、移除 .1 後綴、並使用對應表進行標準化。
    """
    logging.debug("開始標準化欄位名稱...")
    
    # 1. 清理：移除空格並轉為小寫
    original_columns = df.columns
    cleaned_columns = [col.strip().lower() for col in original_columns]
    
    # 2. (新) 移除 '.1', '.2' 等垃圾後綴
    # (例如 'fwd header length.1' -> 'fwd header length')
    # 這必須在 rename 和 drop_duplicates 之前完成
    cleaned_columns = [re.sub(r'\.\d+$', '', col) for col in cleaned_columns]
    df.columns = cleaned_columns
    
    # 3. 移除清理後產生的重複欄位
    # (例如, 現在我們有兩個 'fwd header length', 只保留第一個)
    # 這裡使用 .loc 來避免 SettingWithCopyWarning
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # 4. 重新命名：使用 COLUMN_MAP
    # 找出 df 中存在於對應表 key 的欄位
    renameable_cols = [col for col in df.columns if col in COLUMN_MAP]
    if renameable_cols:
        logging.debug(f"將使用對應表重命名 {len(renameable_cols)} 個欄位...")
        df = df.rename(columns=COLUMN_MAP)
    else:
        logging.warning("欄位名稱對應表中沒有任何欄位與此 DataFrame 的欄位匹配。")

    # 5. 檢查是否有未被對應的欄位 (幫助除錯)
    unmapped_cols = [
        col for col in df.columns 
        if col not in COLUMN_MAP.values() and col not in ['label'] # 排除已是目標或 label
    ]
    if unmapped_cols and 'flow_id' in df.columns: # 只在看起來像特徵檔時警告
        logging.debug(f"注意：有 {len(unmapped_cols)} 個欄位未在對應表中找到 (可能是不需要的欄位): {unmapped_cols[:5]}...") # 只顯示前5個

    return df

def clean_features(df):
    """
    處理 NaN, Infinity，並移除全為 0 的特徵。
    """
    # 處理 Infinity 和 NaN 值
    # 確保只在數字欄位上操作，避免 'label' 欄位出錯
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if df[numeric_cols].isnull().values.any() or \
       np.isinf(df[numeric_cols].values).any():
        
        logging.debug("偵測到 NaN 或 Infinity 值。")
        # 將 Inf 替換為 NaN
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 用 0 填充 NaN (您可以根據需要更改為 .mean() 或 .median())
        # 用 0 填充是 CICFlowMeter 資料集的常見做法
        original_nan_count = df[numeric_cols].isnull().sum().sum()
        if original_nan_count > 0:
            logging.debug(f"正在用 0 填充 {original_nan_count} 個 NaN 值...")
            df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 移除屬性值均為零的特徵
    # 再次僅檢查數字欄位
    numeric_df = df.select_dtypes(include=[np.number])
    all_zero_columns = numeric_df.columns[(numeric_df == 0).all()].tolist()
    
    if all_zero_columns:
        logging.info(f"移除 {len(all_zero_columns)} 個全為 0 的特徵: {all_zero_columns}")
        df = df.drop(columns=all_zero_columns)

    feature_cols = [col for col in FEATURES_WHITELIST if col in df.columns]
    
    missing_in_csv = set(FEATURES_WHITELIST) - set(df.columns)
    if missing_in_csv:
        logging.warning(f"您的 CSV 檔案中缺少 TABLE V 的 {len(missing_in_csv)} 個特徵: {list(missing_in_csv)[:5]}...")
        
    logging.info(f"成功匹配 {len(feature_cols)} / 58 個來自 TABLE V 的特徵。")

    if len(feature_cols) == 0:
        logging.error("沒有找到任何 TABLE V 特徵欄位。請檢查 column_map.py 是否正確。")
        return None, None, None, None, None, None, None

    df = df[feature_cols]
        
    return df

# (新) 定義重採樣的常數
# 必須與 train_stacked_ctgan.py 中的 MAX_SAMPLES_PER_MODEL 一致
TARGET_SAMPLES_PER_CLASS = 15000 
# (*** 新增：定義多數類別的名稱 ***)
MAJORITY_CLASS_NAME = "BENIGN" # 假設 'BENIGN' 已被轉為大寫


def resample_with_hybrid_ctgan(X_train_df: pd.DataFrame, 
                             y_train_series: pd.Series,
                             artifacts_dir: str, 
                             label_encoder,
                             num_samples_per_label: int = 15000):
    """
    (*** 重構：混合採樣 (Hybrid Sampling) ***)
    使用 Stacked-CTGAN 模型來平衡訓練資料集。
    
    - 對 MAJORITY_CLASS_NAME (例如 'BENIGN') 進行「欠採樣」(Undersample) 真實資料。
    - 對所有「其他類別」(攻擊)，載入其 CTGAN 模型並進行「過採樣」(Oversample) 生成合成資料。
    - 對生成的資料進行反向 Log 轉換 (expm1)。
    - 返回一個平衡的、(真實+合成) 混合的資料集。
    """
    logging.info(f"--- 開始「混合採樣」 (目標: {num_samples_per_label} 筆 / 每個類別) ---")
    
    # 將 X 和 y 暫時合併，以便過濾
    df_train = X_train_df.copy()
    df_train['label'] = y_train_series
    
    resampled_dfs = [] # 用於收集所有平衡後的資料框
    unique_labels = y_train_series.unique()

    for label in unique_labels:
        if label == MAJORITY_CLASS_NAME:
            # --- 1. 多數類別：欠採樣 (Undersample) 真實資料 ---
            logging.info(f"  處理多數類別 '{label}': 進行「欠採樣」...")
            label_data = df_train[df_train['label'] == label]
            
            # 確保即使原始資料少於目標，也能正常運作
            sample_size = min(len(label_data), num_samples_per_label)
            resampled_data = label_data.sample(
                n=sample_size, 
                random_state=42
            )
            resampled_dfs.append(resampled_data)
        
        else:
            # --- 2. 少數類別 (攻擊)：過採樣 (Oversample) 合成資料 ---
            model_path = os.path.join(artifacts_dir, f"ctgan_{label}.pkl")
            
            if not os.path.exists(model_path):
                logging.warning(f"  找不到 '{label}' 的 CTGAN 模型。將使用「原始資料」進行欠採樣...")
                # 備案：如果 GAN 不存在，也對其進行欠採樣（或使用全部）
                label_data = df_train[df_train['label'] == label]
                sample_size = min(len(label_data), num_samples_per_label)
                resampled_data = label_data.sample(n=sample_size, random_state=42)
                resampled_dfs.append(resampled_data)
                continue
                
            logging.info(f"  處理少數類別 '{label}': 進行「過採樣」(生成資料)...")
            try:
                label_data = df_train[df_train['label'] == label]
                sample_size = max(0, num_samples_per_label - len(df_train[df_train['label'] == label])) 
                synthetic_data = generate_stacked_data(artifacts_dir=artifacts_dir, label=label, num_samples_per_label=sample_size)
                
                synthetic_data['label'] = label 
                resampled_dfs.append(synthetic_data)
                resampled_dfs.append(label_data)
                
            except Exception as e:
                logging.error(f"  生成 '{label}' 資料失敗: {e}. 跳過此類別。")

    if not resampled_dfs:
        logging.error("未生成或採樣任何資料。中止。")
        return pd.DataFrame(), pd.Series()

    # 將所有處理過的資料框合併
    df_resampled = pd.concat(resampled_dfs, ignore_index=True)
    
    # 徹底隨機打亂最終的資料集
    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"「混合採樣」完成。總共 {len(df_resampled)} 筆資料。")
    
    return df_resampled

def old_data_preprocessing(data_path: str, output_dir: str):
    """
    執行 2017 (D_old) 資料的完整預處理流程：
    1. 載入資料
    2. 標準化欄位名稱 (使用 COLUMN_MAP 並移除 .1)
    3. 清理特徵 (NaN/Inf, 全 0 欄位)
    4. 將標籤轉為大寫並編碼
    5. 分割 7:3 訓練/測試集
    6. 擬合 (fit) 並儲存 MinMaxScaler 和 LabelEncoder
    7. 將資料轉換為 Tensors
    """
    logging.info("--- 開始處理 2017 (D_old) 資料 ---")
    logging.info(f"從 {data_path} 載入資料...")
    try:
        # 嘗試使用 'utf-8' 讀取，如果失敗，嘗試 'latin1'
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            logging.warning("UTF-8 解碼失敗，嘗試使用 'latin1' 編碼...")
            df = pd.read_csv(data_path, encoding='latin1')
            
    except Exception as e:
        logging.error(f"讀取 CSV 失敗: {e}", exc_info=True)
        return None, None, None, None, None, None

    # 1. 標準化欄位名稱 (新步驟)
    df = standardize_columns(df)

    # 2. 清理特徵 (NaN/Inf, 全 0 欄位)
    df = clean_features(df)

    # 3. 分離特徵和標籤
    if 'label' not in df.columns:
        logging.error("資料中找不到 'label' 欄位。請檢查 column_map.py。")
        return None, None, None, None, None, None, None
        
    logging.debug("將標籤轉換為大寫...")
    df['label'] = df['label'].astype(str).str.upper()
    
    y = df['label']
    X = df.drop(columns=['label'])
    
    # (*** 關鍵修正 ***)
    # 我們的特徵集 = 白名單中存在於 X 的欄位

    print (len(X.columns))
    
    # 4. 分割資料 (7:3)
    logging.info("將 2017 資料分割為 70% 訓練集和 30% 測試集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5. (新) 擬合 LabelEncoder (必須在生成資料前)
    #    (重要) 必須在「所有」可能的原始標籤上 fit，以捕獲所有類別
    le = LabelEncoder()
    logging.info("正在使用 2017 *原始* 標籤擬合 (fit) LabelEncoder...")
    le.fit(pd.concat([y_train, y_test], ignore_index=True)) 


    # 7. 擬合 (Fit) Scaler 和 Encoder
    scaler = MinMaxScaler()
    logging.info("正在使用 2017 (CTGAN 處理後) 訓練資料擬合 (fit) MinMaxScaler...")
    scaler.fit(X_train) # (新) 在 resampled 資料上 fit
    
    le = LabelEncoder()
    logging.info("正在使用 2017 訓練標籤擬合 (fit) LabelEncoder...")
    le.fit(y_train) # (不變) 仍然在「原始」y_train 上 fit，以學習真實標籤

    # 7. (*** 新 ***) 計算類別權重 (Class Weights)
    logging.info("正在計算類別權重以處理不平衡問題...")
    y_train_encoded = le.transform(y_train) # 轉換為數字
    
    class_counts_encoded = pd.Series(y_train_encoded).value_counts()
    class_counts_df = pd.DataFrame({'counts': class_counts_encoded})
    # .reindex 確保權重向量的索引 (0, 1, 2...) 與 le.classes_ 的順序一致
    class_counts_sorted = class_counts_df.reindex(range(len(le.classes_))).fillna(1)['counts']
    
    total_samples = len(y_train_encoded)
    num_classes = len(le.classes_)
    
    weights_series = total_samples / (num_classes * class_counts_sorted)
    
    # (*** 關鍵修正 ***)
    # 裁切 (Clip) 權重，避免極端值。
    weights_series = weights_series.clip(lower=1.0, upper=8.2)
    logging.info("權重已被裁切 (Clip) 在 [1.0, 10.0] 範圍內。")

    class_weights_tensor = torch.tensor(weights_series.values, dtype=torch.float32)
    
    try:
        benign_index = le.transform(['BENIGN'])[0]
        logging.info(f"類別權重計算完成。例如 'BENIGN' (索引 {benign_index}): {weights_series[benign_index]:.4f}")
    except ValueError:
        logging.warning("在 y_train 中找不到 'BENIGN' 標籤來顯示權重範例。")
    
    # 8. 儲存 Artifacts
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "minmax_scaler_2017.joblib")
    le_path = os.path.join(output_dir, "label_encoder_2017.joblib")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    logging.info(f"MinMaxScaler (包含 {len(scaler.feature_names_in_)} 個特徵) 已儲存至: {scaler_path}")
    logging.info(f"LabelEncoder (包含 {len(le.classes_)} 個類別) 已儲存至: {le_path}")

    # 9. 轉換 (Transform) 資料以供回傳
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)

    input_features = X_train_scaled.shape[1]
    num_classes = len(le.classes_)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    # (新) 返回 7 個項目
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, \
           input_features, num_classes, le, class_weights_tensor 

def preprocess_new_data(data_path: str, artifacts_dir: str):
    """
    使用 2017 年儲存的 artifacts 來處理 2018 (D_new) 資料：
    1. 載入資料
    2. 標準化欄位名稱 (使用 COLUMN_MAP 並移除 .1)
    3. 清理特徵 (NaN/Inf, 全 0 欄位)
    4. 載入 2017 年的 Scaler 和 Encoder
    5. 將標籤轉為大寫
    6. 過濾 2018 資料，只保留 2017 Encoder 中已知的標籤
    7. 對齊特徵欄位，只使用 2017 Scaler 中已知的特徵
    8. 轉換 (Transform) 資料
    9. 將資料轉換為 Tensors
    """
    logging.info("--- 開始處理 2018 (D_new) 資料 ---")
    
    # 1. 載入 Artifacts
    scaler_path = os.path.join(artifacts_dir, "minmax_scaler_2017.joblib")
    le_path = os.path.join(artifacts_dir, "label_encoder_2017.joblib")
    
    try:
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        logging.info("成功載入 2017 年的 Scaler 和 LabelEncoder。")
    except FileNotFoundError:
        logging.error("找不到 2017 年的 artifact 檔案。請先執行 2017 年資料的訓練。")
        return None, None, None
    except Exception as e:
        logging.error(f"載入 artifacts 失敗: {e}", exc_info=True)
        return None, None, None

    # 2. 載入 2018 資料
    logging.info(f"從 {data_path} 載入 2018 資料...")
    try:
        # 嘗試使用 'utf-8' 讀取，如果失敗，嘗試 'latin1'
        try:
            df_2018 = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            logging.warning("UTF-8 解碼失敗，嘗試使用 'latin1' 編碼...")
            df_2018 = pd.read_csv(data_path, encoding='latin1')
            
    except Exception as e:
        logging.error(f"讀取 2018 CSV 失敗: {e}", exc_info=True)
        return None, None, None
        
    # 3. 標準化 2018 欄位名稱
    df_2018 = standardize_columns(df_2018)

    # 4. 清理 2018 特徵
    df_2018 = clean_features(df_2018)

    # 5. 將標籤轉為大寫以進行比對
    if 'label' not in df_2018.columns:
        logging.error("2018 資料中找不到 'label' 欄位。")
        return None, None, None
        
    df_2018['label'] = df_2018['label'].astype(str).str.upper()
    df_2018['label'] = df_2018['label'].replace('FTP-BRUTEFORCE', 'FTP-PATATOR')
    df_2018['label'] = df_2018['label'].replace('SSH-BRUTEFORCE', 'SSH-PATATOR')

    # 6. 過濾標籤
    # 只保留 2017 年 LabelEncoder 知道的標籤
    known_labels = set(le.classes_)
    original_count = len(df_2018)
    df_2018_filtered = df_2018[df_2018['label'].isin(known_labels)]
    filtered_count = len(df_2018_filtered)
    
    if filtered_count == 0:
        logging.warning("2018 年資料中未包含任何 2017 年已知的標籤。無法評估。")
        return None, None, None
    
    logging.info(f"標籤過濾：保留了 {filtered_count} / {original_count} 筆資料 (只保留 2017 年已知的標籤)。")

    # 7. 對齊特徵欄位
    try:
        # 從 scaler 獲取 2017 年的標準特徵列表
        feature_names_2017 = scaler.feature_names_in_
        
        # 檢查 2018 資料是否包含所有必要的特徵
        missing_cols = set(feature_names_2017) - set(df_2018_filtered.columns)
        if missing_cols:
            logging.error(f"2018 年資料中缺少必要的特徵: {missing_cols}")
            return None, None, None
            
        # 按 2017 年的順序和列表篩選 2018 年的特徵
        features_2018 = df_2018_filtered[feature_names_2017]
        labels_2018 = df_2018_filtered['label']
        
    except KeyError as e:
        logging.error(f"欄位對齊失敗: {e}。這通常意味著 2018 年的資料在標準化後仍缺少 2017 年模型所需的欄位。")
        return None, None, None
    except AttributeError:
        logging.error("載入的 Scaler 物件沒有 'feature_names_in_' 屬性。請確保 Scaler 是用 Pandas DataFrame 擬合的。")
        return None, None, None

    # 8. 轉換 (Transform) 資料
    logging.info("正在使用 2017 年的 Scaler 和 Encoder 轉換 2018 年資料...")
    X_2018_scaled = scaler.transform(features_2018)
    y_2018_encoded = le.transform(labels_2018)
    
    # 9. 將資料轉換為 Tensors
    X_test_tensor = torch.tensor(X_2018_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_2018_encoded, dtype=torch.long)

    logging.info("2018 (D_new) 資料處理完成。")
    return X_test_tensor, y_test_tensor, le

# (新) 用於 A-NIDS 重新訓練的輔助函式
def load_and_clean_data(data_path: str, artifacts_dir: str):
    """
    (用於「訓練」)
    載入 2018 (D_new) 資料，清理並標準化欄位，
    返回「未歸一化」的 DataFrame (X) 和 Series (y)，
    僅保留 2017 年已知的特徵和標籤。
    """
    logging.info(f"--- (訓練模式) 載入並清理 {data_path} ---")
    
    # 1. 載入 Artifacts (僅用於獲取特徵和標籤列表)
    scaler_path = os.path.join(artifacts_dir, "minmax_scaler_2017.joblib")
    le_path = os.path.join(artifacts_dir, "label_encoder_2017.joblib")
    try:
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        feature_names_2017 = scaler.feature_names_in_
        known_labels = set(le.classes_)
    except Exception as e:
        logging.error(f"載入 artifacts 失敗: {e}", exc_info=True)
        return None, None

    # 2. 載入 2018 資料
    try:
        try:
            df_2018 = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_2018 = pd.read_csv(data_path, encoding='latin1')
    except Exception as e:
        logging.error(f"讀取 2018 CSV 失敗: {e}", exc_info=True)
        return None, None
        
    # 3. 標準化欄位 & 清理特徵
    df_2018 = standardize_columns(df_2018)
    df_2018 = clean_features(df_2018)
    
    if 'label' not in df_2018.columns:
        logging.error("2018 資料中找不到 'label' 欄位。")
        return None, None
    df_2018['label'] = df_2018['label'].astype(str).str.upper()
    df_2018['label'] = df_2018['label'].replace('FTP-BRUTEFORCE', 'FTP-PATATOR')
    df_2018['label'] = df_2018['label'].replace('SSH-BRUTEFORCE', 'SSH-PATATOR')

    # 4. 過濾標籤和特徵
    df_2018_filtered = df_2018[df_2018['label'].isin(known_labels)]
    
    missing_cols = set(feature_names_2017) - set(df_2018_filtered.columns)
    if missing_cols:
        logging.error(f"2018 年資料中缺少必要的特徵: {missing_cols}")
        return None, None
        
    # 按 2017 年的順序和列表篩選
    X_new_real = df_2018_filtered[feature_names_2017]
    y_new_real = df_2018_filtered['label']
    
    logging.info(f"成功載入並清理了 {len(X_new_real)} 筆 2018 (D_new) 真實訓練資料。")
    return X_new_real, y_new_real