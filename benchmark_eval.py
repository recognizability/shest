import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os

def dice_score(y_true, y_pred, class_idx):
    y_true_bin = (y_true == class_idx).astype(int)
    y_pred_bin = (y_pred == class_idx).astype(int)
    intersection = np.sum(y_true_bin * y_pred_bin)
    return 2 * intersection / (np.sum(y_true_bin) + np.sum(y_pred_bin) + 1e-6)

def evaluate_and_save_benchmark(y_true, y_pred, class_names, save_path):
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    results = {
        "Accuracy": accuracy_score(y_true_flat, y_pred_flat),
        "Macro Precision": precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
        "Macro Recall": recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
        "Macro F1-score": f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0),
    }

    per_class_f1 = f1_score(y_true_flat, y_pred_flat, average=None, labels=range(len(class_names)), zero_division=0)
    for name, f1 in zip(class_names, per_class_f1):
        results[f"F1 ({name})"] = f1

    for i, name in enumerate(class_names):
        dice = dice_score(y_true_flat, y_pred_flat, i)
        results[f"Dice ({name})"] = dice

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
    df.to_csv(save_path, index=False)
    return df

# Display usage steps for user
import ace_tools as tools; tools.display_dataframe_to_user(name="SHEST Benchmark Script (Template)", dataframe=pd.DataFrame({
    "Step": ["1. Call `evaluate_and_save_benchmark(...)`", 
             "2. Pass true/pred labels & class names", 
             "3. Get scores (F1 + Dice)", 
             "4. Save results to CSV"],
    "Example": ["evaluate_and_save_benchmark(...)",
                "y_true, y_pred, [class names]",
                "F1 (macro, per class), Dice",
                "./results/benchmark.csv"]
}))