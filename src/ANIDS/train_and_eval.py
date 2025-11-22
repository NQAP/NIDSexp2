# train_and_eval.py
#
# (新) 新增 FocalLoss 類別 (論文 [25] 的方法)。
# (新) train_model 現在增加 `use_focal_loss` 參數來切換損失函式。
# (新) 修正 plot_training_history 的 `plot_filename` 參數。

import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm # 顯示進度條
from utils import set_seed

# 檢查是否有可用的 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"將使用 {DEVICE} 設備進行訓練和評估。")

# -------------------------------------------------------------------
# (*** 新 ***) Focal Loss 的 PyTorch 實作
# -------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss (焦點損失) - 實現「動態權重更新」
    專門用於解決類別不平衡和困難樣本的損失函式。
    """
    def __init__(self, alpha: torch.Tensor = None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Alpha 權重 (可選)
        self.gamma = gamma # Gamma (gamma=0 時等同於 CrossEntropy)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): 模型的原始輸出 (logits)，形狀 (N, C)
            targets (torch.Tensor): 真實標籤 (索引)，形狀 (N)
        """
        
        # 1. 計算標準的 CrossEntropy Loss，但不進行 reduction
        # ce_loss 已經是 -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 2. 計算 p_t (模型對「正確答案」的預測機率)
        log_pt = F.log_softmax(inputs, dim=1).gather(1, targets.view(-1, 1)).squeeze(1)
        pt = torch.exp(log_pt)

        # 3. 計算「動態」專注權重 (gamma)
        # (1-pt)^gamma
        # 如果 pt 很高 (例如 0.99, 簡單樣本)，權重 -> (0.01)^2 = 0.0001 (權重降低)
        # 如果 pt 很低 (例如 0.1, 困難樣本)，權重 -> (0.9)^2 = 0.81 (權重保持)
        focusing_factor = torch.pow((1.0 - pt), self.gamma)
        
        # 4. (*** 新 ***) 獲取「靜態」類別權重 (alpha)
        if self.alpha is not None:
            # 確保 alpha tensor 在同一個設備上
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            
            # 根據 targets (標籤索引) 查找對應的 class_weight
            # targets.data.view(-1) -> [0, 2, 1, 0, ...]
            # self.alpha.gather(0, ...) -> [w_0, w_2, w_1, w_0, ...]
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
        else:
            alpha_t = 1.0 # 如果沒有提供權重，則 alpha 為 1

        # 5. 應用「兩種」權重
        # Loss = alpha * (1-pt)^gamma * ce_loss
        focal_loss = alpha_t * focusing_factor * ce_loss

        # 6. 應用 Reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
# -------------------------------------------------------------------


def train_model(model: nn.Module, 
                X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor, # 驗證集
                label_encoder: LabelEncoder, # (*** 新 ***)
                epochs: int, 
                batch_size: int, 
                learning_rate: float,
                class_weights: torch.Tensor = None,
                use_focal_loss: bool = True,
                gamma: float = 0.0): # <-- (*** 新 ***)
    """
    訓練 FCN 模型。
    (新) 接受 class_weights 參數以處理類別不平衡。
    (新) 接受 label_encoder 並在每 10 個 epochs 顯示報告。
    """
    set_seed(42)
    model.to(DEVICE)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # (*** 新 ***) 根據情境選擇損失函式
    if class_weights is not None and use_focal_loss:
        # Phase 1: 訓練 Mlp-2017 (資料不平衡)
        criterion = FocalLoss(alpha=class_weights, gamma=gamma).to(DEVICE) # <-- (新) 使用傳入的 gamma
        logging.info("使用加權交叉熵損失 (Weighted Cross-Entropy) 來處理類別不平衡。")
    
    # (*** 新 ***) 根據情境選擇損失函式
    elif use_focal_loss:
        # Phase 1: 訓練 Mlp-2017 (資料不平衡)
        criterion = FocalLoss(gamma=gamma).to(DEVICE) # <-- (新) 使用傳入的 gamma
        logging.info(f"使用 Focal Loss (gamma={gamma}) 來處理 Phase 1 的類別不平衡。")

    else:
        # Phase 3: 訓練 A-NIDS (資料已手動平衡)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        logging.info("使用標準 (未加權) 損失函式 (適用於已平衡的 Phase 3 訓練)。")
    

    # (*** 新 ***)
    # 加入 L2 正則化 (weight_decay) 來對抗過擬合
    optimizer = optim.Adam(model.parameters(), 
                           lr=learning_rate, 
                           weight_decay=1e-5) 
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    logging.info(f"--- 開始訓練 FCN (Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}, WeightDecay: 1e-5) ---")
    
    for epoch in range(epochs):
        model.train() 
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)
        
        # --- 執行驗證 ---
        model.eval() 
        running_val_loss = 0.0
        
        # (*** 新 ***) 收集所有預測和標籤
        all_val_preds = []
        all_val_labels = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        val_criterion = nn.CrossEntropyLoss().to(DEVICE)
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = val_criterion(outputs, labels) 
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        # (*** 新 ***) 在迴圈外計算總損失和總準確率
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = 100 * accuracy_score(all_val_labels, all_val_preds)
        
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        
        logging.info(f"Epoch {epoch+1}/{epochs} - "
                     f"Train Loss: {epoch_train_loss:.4f}, "
                     f"Val Loss: {epoch_val_loss:.4f}, "
                     f"Val Acc: {epoch_val_accuracy:.2f}%")
        
        # (*** 新 ***) 每 10 個 Epochs 或最後一個 Epoch，顯示矩陣
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            try:
                class_names = label_encoder.classes_
                cm = confusion_matrix(all_val_labels, all_val_preds, labels=range(len(class_names)))
                report = classification_report(all_val_labels, all_val_preds, target_names=class_names, zero_division=0)
                
                logging.info(f"\n--- Validation Confusion Matrix (Epoch {epoch+1}) ---\n{cm}")
                logging.info(f"\n--- Validation Report (Epoch {epoch+1}) ---\n{report}")
            except Exception as e:
                logging.warning(f"無法在 epoch {epoch+1} 生成驗證報告: {e}")

    logging.info("--- FCN 訓練完成 ---")
    return model, history

def evaluate_model(model: nn.Module, 
                   X_test: torch.Tensor, y_test: torch.Tensor, 
                   label_encoder: LabelEncoder,
                   output_dir: str,
                   dataset_name: str):
    """
    在測試集上評估模型效能，並儲存報告和混淆矩陣。
    """
    
    logging.info(f"--- 開始評估模型於: {dataset_name} ---")
    
    model.to(DEVICE)
    model.eval()
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)
    
    all_preds = []
    all_labels = []

    logging.info("正在進行預測...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="[Evaluate]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    try:
        class_names = label_encoder.classes_
    except AttributeError:
        logging.error("LabelEncoder 格式錯誤。")
        class_names = [str(i) for i in range(len(np.unique(all_labels)))]

    # --- 1. 計算準確率 ---
    accuracy = 100 * accuracy_score(all_labels, all_preds)
    logging.info(f"整體準確率 (Accuracy) on {dataset_name}: {accuracy:.2f}%")

    # --- 2. 儲存分類報告 (Classification Report) 為 .csv ---
    logging.info("正在生成分類報告...")
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print("分類報告：")
    print(report_df)
    report_path = os.path.join(output_dir, f"report_{dataset_name}.csv")
    report_df.to_csv(report_path)
    logging.info(f"分類報告已儲存至: {report_path}")

    # --- 3. 繪製並儲存混淆矩陣 (Confusion Matrix) 為 .png ---
    logging.info("正在生成混淆矩陣圖...")
    cm = confusion_matrix(all_labels, all_preds)
    
    num_classes = len(class_names)
    fig_width = max(10, num_classes * 0.8)
    fig_height = max(8, num_classes * 0.6)
        
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout() 
    
    cm_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name}.png")
    plt.savefig(cm_path)
    plt.close() 
    logging.info(f"混淆矩陣圖已儲存至: {cm_path}")
    logging.info(f"--- 評估 {dataset_name} 完成 ---")
    
def plot_training_history(history: dict, output_dir: str, plot_filename: str = "training_history.png"):
    """
    繪製訓練過程中的損失和準確率曲線。
    (新) 修正：現在可以正確接收 plot_filename 參數。
    """
    logging.info(f"正在繪製訓練歷史圖 ({plot_filename})...")
    
    try:
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # 圖 1: 訓練損失 vs 驗證損失
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Training Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 圖 2: 驗證準確率
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        
        # (新) 使用傳入的 plot_filename
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"訓練歷史圖已儲存至: {plot_path}")
        
    except Exception as e:
        logging.error(f"繪製訓練圖表失敗: {e}", exc_info=True)