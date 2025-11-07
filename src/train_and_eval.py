import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    """
    訓練偵測模型 (Mlp-2017) 的函式。
    現在包含驗證迴圈，並返回 history 用於繪圖。
    """
    logging.info(f"開始模型訓練... (Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate})")
    
    # 建立 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2) # 驗證時 batch size 可以較大
    
    # 定義損失函式和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 用於儲存歷史記錄
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        # --- 訓練迴圈 ---
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)
        
        # --- 驗證迴圈 ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = 100 * correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        # 記錄每個 epoch 的損失
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
             logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                          f"Train Loss: {epoch_train_loss:.6f} | "
                          f"Val Loss: {epoch_val_loss:.6f} | "
                          f"Val Acc: {epoch_val_accuracy:.2f}%")
            
    logging.info("模型訓練完成。")
    return model, history

def evaluate_model(model, X_test, y_test, label_encoder, output_dir: str, dataset_name: str = "測試集"):
    """
    評估模型效能，現在會輸出完整的 classification_report 和 confusion_matrix。
    """
    logging.info(f"--- 開始在 ({dataset_name}) 上進行詳細評估 ---")
    model.eval() # 設置為評估模式
    
    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 從 LabelEncoder 獲取類別名稱
    try:
        class_names = label_encoder.classes_
    except AttributeError:
        logging.warning("無法從 label_encoder 獲取類別名稱，將使用索引。")
        class_names = [str(i) for i in range(len(set(all_labels)))]

    # 計算並記錄主要指標
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_w = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall_w = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1_w = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    logging.info(f"({dataset_name}) - 整體效能 (Weighted Avg):")
    logging.info(f"  Accuracy:    {accuracy * 100:.2f}%")
    logging.info(f"  Precision:   {precision_w:.4f}")
    logging.info(f"  Recall:      {recall_w:.4f}")
    logging.info(f"  F1-Score:    {f1_w:.4f}")
    
    # 記錄 Classification Report
    logging.info(f"({dataset_name}) - 分類報告 (Classification Report):")
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    print("\n" + report + "\n") # 直接 print 以保持格式

    # 記錄 Confusion Matrix
    logging.info(f"({dataset_name}) - 混淆矩陣 (Confusion Matrix):")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    # --- (新) 儲存報告和矩陣到檔案 ---
    try:
        # 建立一個安全的檔案名稱
        safe_filename = dataset_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')
        
        # --- (新) 儲存分類報告為 CSV ---
        report_dict = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_path = os.path.join(output_dir, f"report_{safe_filename}.csv")
        report_df.to_csv(report_path)
        logging.info(f"分類報告 (CSV) 已儲存至: {report_path}")

        # --- (新) 儲存混淆矩陣為熱圖 (PNG) ---
        cm_plot_path = os.path.join(output_dir, f"confusion_matrix_{safe_filename}.png")
        
        # 根據類別數量動態調整圖表大小
        num_classes = len(class_names)
        fig_width = max(10, num_classes * 0.5)
        fig_height = max(8, num_classes * 0.4)
        
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, 
                    annot=True, # 在格子中顯示數字
                    fmt='d',      # 數字格式為整數
                    cmap='Blues', # 顏色主題
                    xticklabels=class_names, 
                    yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - {dataset_name}', fontsize=16)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout() # 自動調整佈局
        
        plt.savefig(cm_plot_path)
        plt.close() # 關閉圖表以釋放記憶體
        logging.info(f"混淆矩陣 (PNG) 已儲存至: {cm_plot_path}")

    except Exception as e:
        logging.error(f"儲存評估報告失敗: {e}", exc_info=True)

    
    print("-" * (len(dataset_name) + 30) + "\n")

    return accuracy

def plot_training_history(history, output_dir, plot_filename):
    """
    使用 matplotlib 繪製訓練和驗證歷史，並儲存圖檔。
    """
    logging.info("正在繪製訓練歷史圖表...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Training and Validation History', fontsize=16)

    # 繪製 損失 (Loss)
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs. Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 繪製 準確率 (Accuracy)
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # 儲存圖檔
    plot_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_path)
        logging.info(f"訓練歷史圖表已儲存至: {plot_path}")
    except Exception as e:
        logging.error(f"儲存圖表失敗: {e}")