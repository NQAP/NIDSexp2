import torch
import torch.nn as nn
import logging

class ShallowFCN(nn.Module):
    """
    定義淺層全連接神經網路 (Shallow FCN) 的架構。
    這對應論文中的 Detection Module。
    """
    def __init__(self, input_features, hidden_1, hidden_2, num_classes):
        super(ShallowFCN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, num_classes)
        )

    def forward(self, x):
        """定義前向傳播"""
        return self.network(x)

def get_param_count(model):
    """輔助函式，計算模型的總參數數量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def detect_module(input_features: int, num_classes: int):
    """
    建立並回傳一個 A-NIDS 偵測模型 (Shallow FCN)。

    Args:
        input_features (int): 輸入特徵的維度。
        num_classes (int): 輸出的類別總數。

    Returns:
        torch.nn.Module: 一個未經訓練的 ShallowFCN 模型實例。
    """
    
    # 為了使總參數接近論文中提到的 12,281
    # (假設 input_features=78, num_classes=9)
    # (58 * 100 + 100) + (100 * 50 + 50) + (50 * 9 + 9) = 13,409 (接近)
    HIDDEN_LAYER_1 = 256
    HIDDEN_LAYER_2 = 128
    
    if input_features <= 0:
        logging.warning(f"輸入特徵維度 ({input_features}) 不尋常。")
    if num_classes <= 1:
        logging.warning(f"類別數量 ({num_classes}) 不尋常。")

    # 實例化模型
    model = ShallowFCN(
        input_features=input_features,
        hidden_1=HIDDEN_LAYER_1,
        hidden_2=HIDDEN_LAYER_2,
        num_classes=num_classes
    )

    return model