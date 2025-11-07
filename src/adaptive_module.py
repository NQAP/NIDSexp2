# adaptive_module.py
#
# 封裝 A-NIDS 適應性模組 (Clustering) 的所有邏輯。
# 根據論文 V-C 節，我們使用 K-Means 來無監督地偵測資料漂移。

import os
import joblib
import logging
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def train_adaptive_model(X_scaled_old: np.ndarray, 
                        k_clusters: int, 
                        alpha: float, 
                        or_threshold: float, 
                        output_dir: str):
    """
    訓練 K-Means 聚類模型 (f_C)，並計算 D_old 的信賴區間 (CI) 和基礎離群率。
    
    Args:
        X_scaled_old (np.ndarray): 完整的 2017 (D_old) **歸一化**特徵資料。
        k_clusters (int): K-Means 的叢集數量 (論文中的 K)。
        alpha (float): 計算 CI 的信賴係數 (論文中的 α)。
        or_threshold (float): 觸發漂移的離群值比率 (OR) 閾值 (論文中的 T)。
        output_dir (str): 儲存 artifacts 的目錄。
    """
    
    logging.info("--- 開始訓練 Adaptive Module (K-Means) ---")
    logging.info(f"使用 K={k_clusters}, α={alpha}, T(OR Threshold)={or_threshold}")
    
    # --- 1. 訓練 K-Means 模型 ---
    # 論文 [cite: 500] 建議 K 遠大於真實標籤數 (例如 K=40 或 50)
    # 為了加速，我們這裡使用 K=k_clusters
    logging.info(f"在 {X_scaled_old.shape[0]} 筆 D_old 資料上訓練 K-Means (K={k_clusters})...")
    
    # 為了效能，使用 MiniBatchKMeans，效果與 KMeans 接近
    # n_init='auto' 會自動處理，以獲得最佳效能
    kmeans_model = KMeans(
        n_clusters=k_clusters, 
        random_state=42, 
        n_init='auto',
        max_iter=300
    )
    kmeans_model.fit(X_scaled_old)
    
    model_path = os.path.join(output_dir, "kmeans_model.joblib")
    joblib.dump(kmeans_model, model_path)
    logging.info(f"K-Means 模型已儲存至: {model_path}")

    # --- 2. 計算 D_old 的距離和信賴區間 (CI) ---
    logging.info("計算 D_old 樣本到其叢集中心的距離...")
    # L(X_j) = ||X_j - C(Y_Xj)||^2 [cite: 307]
    # pairwise_distances_argmin_min 返回最小距離的平方根，我們需要平方它們
    # 修正：直接使用 .transform() 獲取到所有中心的距離，然後取最小值
    distances_old = kmeans_model.transform(X_scaled_old).min(axis=1)
    
    # CI = E[L(X_j)] + α * S[L(X_j)] [cite: 311]
    mean_dist_old = np.mean(distances_old)
    std_dist_old = np.std(distances_old)
    ci_threshold = mean_dist_old + (alpha * std_dist_old)
    
    logging.info(f"D_old 距離均值 (E): {mean_dist_old:.6f}")
    logging.info(f"D_old 距離標準差 (S): {std_dist_old:.6f}")
    logging.info(f"計算出的信賴區間 (CI) 閾值: {ci_threshold:.6f}")

    # --- 3. 計算 D_old 的基礎離群率 (Base Outlier Ratio) ---
    outliers_old_count = np.sum(distances_old > ci_threshold)
    base_outlier_ratio = outliers_old_count / len(X_scaled_old)
    
    if base_outlier_ratio == 0:
        logging.warning("D_old 的基礎離群率為 0.0。這可能導致 OR 偵測過於敏感。")
        # 設置一個極小值以避免除以零
        base_outlier_ratio = 1e-9 

    logging.info(f"D_old 基礎離群值: {outliers_old_count} / {len(X_scaled_old)} (Ratio: {base_outlier_ratio:.6f})")

    # --- 4. 儲存設定檔 ---
    adaptive_config = {
        'k_clusters': k_clusters,
        'alpha': alpha,
        'ci_threshold': ci_threshold,
        'base_outlier_ratio': base_outlier_ratio,
        'or_threshold_T': or_threshold
    }
    
    config_path = os.path.join(output_dir, "adaptive_config.json")
    with open(config_path, 'w') as f:
        json.dump(adaptive_config, f, indent=4)
        
    logging.info(f"Adaptive Module 設定已儲存至: {config_path}")
    logging.info("--- Adaptive Module 訓練完成 ---")


def check_for_drift(X_scaled_new: np.ndarray, artifacts_dir: str) -> bool:
    """
    載入已訓練的 K-Means 模型和設定，檢查新資料 (D_new) 是否觸發資料漂移。
    
    Args:
        X_scaled_new (np.ndarray): 新的**歸一化**資料窗口。
        artifacts_dir (str): 儲存 K-Means 模型和 config.json 的目錄。
        
    Returns:
        bool: True 表示偵測到漂移，False 表示未偵測到。
    """
    
    logging.debug(f"--- (Adaptive Check) 檢查 {len(X_scaled_new)} 筆新資料 ---")
    
    # --- 1. 載入 Artifacts ---
    try:
        model_path = os.path.join(artifacts_dir, "kmeans_model.joblib")
        config_path = os.path.join(artifacts_dir, "adaptive_config.json")
        
        kmeans_model = joblib.load(model_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    except FileNotFoundError:
        logging.error("Adaptive Module 尚未訓練 (找不到 artifacts)！請先執行 train_adaptive_main.py。")
        return False # 無法檢查

    # --- 2. 獲取設定 ---
    ci_threshold = config['ci_threshold']
    base_outlier_ratio = config['base_outlier_ratio']
    or_threshold_T = config['or_threshold_T']

    # --- 3. 計算 D_new 的離群率 ---
    # 使用 D_old 的 K-Means 模型計算 D_new 的距離
    distances_new = kmeans_model.transform(X_scaled_new).min(axis=1)
    
    outliers_new_count = np.sum(distances_new > ci_threshold)
    new_outlier_ratio = outliers_new_count / len(X_scaled_new)

    # --- 4. 計算 OR (Outlier Ratio) ---
    # OR = (Outlier % in D_new) / (Outlier % in D_old) 
    if base_outlier_ratio == 0:
        base_outlier_ratio = 1e-9 # 避免除以零
        
    OR = new_outlier_ratio / base_outlier_ratio
    
    logging.debug(f"D_new 離群率: {new_outlier_ratio:.6f} | D_old 基礎: {base_outlier_ratio:.6f}")
    logging.info(f"計算出的離群值比率 (OR): {OR:.4f}")

    # --- 5. 比較閾值 T ---
    if OR > or_threshold_T:
        logging.warning(f"[!! DATA DRIFT DETECTED !!] (OR: {OR:.4f} > Threshold T: {or_threshold_T})")
        return True
    else:
        logging.info(f"[No Drift] (OR: {OR:.4f} <= Threshold T: {or_threshold_T})")
        return False