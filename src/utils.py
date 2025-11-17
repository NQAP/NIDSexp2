import random
import numpy as np
import torch
import logging
import os

def set_seed(seed: int = 42):
    """
    設定所有隨機種子以確保可重現性 (Reproducibility)。
    """
    # 1. Python 原生的 random
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    
    # 4. PyTorch 在 GPU 上的相關設定
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if multi-GPU
        
        # (*** 關鍵 ***) 
        # 關閉 cuDNN 的 benchmark 模式，
        # 並啟用 deterministic 模式來確保 GPU 上的可重現性
        # 這可能會讓訓練「稍微」變慢，但是是確保結果一致的必要犧牲
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    logging.info(f"已將所有隨機種子 (random, numpy, torch, cuDNN) 設定為 {seed}")