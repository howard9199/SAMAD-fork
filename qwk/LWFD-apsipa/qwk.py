import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.special import kl_div

def quadratic_kappa_coefficient(output, target):
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum() # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK

def to_onehot(tensor, num_classes=5):
    # 確保輸入值在 1-5 範圍內
    tensor = torch.clamp(tensor, 1, 5)
    # 將 1-5 轉換為 0-4 的索引
    tensor = tensor - 1
    # 轉換為 one-hot 向量
    return F.one_hot(tensor.long(), num_classes=num_classes).float()

def calculate_accuracy(output, target):
    # 將 one-hot 向量轉回類別索引
    pred = torch.argmax(output, dim=1) + 1  # +1 轉回 1-5 分數
    true = torch.argmax(target, dim=1) + 1
    # 計算準確率
    correct = (pred == true).sum().item()
    total = len(true)
    return correct / total

def calculate_mse(output, target):
    # 轉換回原始分數
    pred = torch.argmax(output, dim=1) + 1
    true = torch.argmax(target, dim=1) + 1
    return F.mse_loss(pred.float(), true.float()).item()

def plot_confusion_matrix(output, target):
    # 轉換回原始分數
    pred = torch.argmax(output, dim=1) + 1
    true = torch.argmax(target, dim=1) + 1
    
    # 計算混淆矩陣
    cm = confusion_matrix(true.numpy(), pred.numpy())
    
    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return cm

def calculate_distribution_similarity(pred_counts, true_counts):
    # 轉換為numpy array並正規化
    pred_dist = pred_counts.numpy() / pred_counts.sum()
    true_dist = true_counts.numpy() / true_counts.sum()
    
    # 1. 計算 Cosine Similarity
    cosine_sim = F.cosine_similarity(torch.tensor(pred_dist).unsqueeze(0), 
                                   torch.tensor(true_dist).unsqueeze(0)).item()
    
    # 2. 計算 Pearson Correlation
    pearson_corr, _ = pearsonr(pred_dist, true_dist)
    
    return {
        'cosine_similarity': cosine_sim,
        'pearson_correlation': pearson_corr
    }

def plot_score_distribution(output, target):
    # 轉換回原始分數
    pred = torch.argmax(output, dim=1) + 1
    true = torch.argmax(target, dim=1) + 1
    
    # 計算各分數的數量
    pred_counts = torch.bincount(pred, minlength=6)[1:]  # 排除0，只要1-5
    true_counts = torch.bincount(true, minlength=6)[1:]
    
    # 計算分佈相似度
    similarities = calculate_distribution_similarity(pred_counts, true_counts)
    
    # 繪製折線圖
    plt.figure(figsize=(10, 6))
    scores = range(1, 6)
    plt.plot(scores, pred_counts.numpy(), 'b-o', label='Predicted')
    plt.plot(scores, true_counts.numpy(), 'r-o', label='True')
    
    # 在圖上添加相似度資訊
    plt.text(0.02, 0.98, f'Cosine Similarity: {similarities["cosine_similarity"]:.4f}\n'
                        f'Pearson Correlation: {similarities["pearson_correlation"]:.4f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    plt.xticks(scores)
    plt.savefig('score_distribution.png')
    plt.close()
    
    return similarities

def calculate_class_accuracy(output, target):
    # 轉換回原始分數
    pred = torch.argmax(output, dim=1) + 1
    true = torch.argmax(target, dim=1) + 1
    
    class_accuracies = {}
    for score in range(1, 6):
        # 找出真實標籤為該分數的樣本
        mask = (true == score)
        if mask.sum() > 0:  # 確保有該分數的樣本
            # 計算該分數的正確率
            correct = ((pred == score) & mask).sum().item()
            total = mask.sum().item()
            accuracy = correct / total
            class_accuracies[score] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
    
    return class_accuracies

def main():
    parser = argparse.ArgumentParser(description='Calculate Quadratic Weighted Kappa')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file containing output and target arrays')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output array')
    parser.add_argument('-t', '--target', type=str, required=True, help='Target array')
    args = parser.parse_args()

    data = pd.read_csv(args.file)
    output = torch.tensor(data[args.output])
    target = torch.tensor(data[args.target])
    
    # 轉換為 one-hot 向量
    output = to_onehot(output)
    target = to_onehot(target)
    print(output.shape, target.shape)

    qwk = quadratic_kappa_coefficient(output, target)
    acc = calculate_accuracy(output, target)
    mse = calculate_mse(output, target)
    similarities = plot_score_distribution(output, target)
    class_accuracies = calculate_class_accuracy(output, target)
    
    print(f'Quadratic Weighted Kappa: {qwk.item()}')
    print(f'Accuracy: {acc:.4f}')
    print(f'MSE: {mse:.4f}')
    print('\nDistribution Similarities:')
    print(f'Cosine Similarity: {similarities["cosine_similarity"]:.4f}')
    print(f'Pearson Correlation: {similarities["pearson_correlation"]:.4f}')
    
    print('\nPer-class Accuracies:')
    for score, stats in class_accuracies.items():
        print(f'Score {score}: {stats["accuracy"]:.4f} ({stats["correct"]}/{stats["total"]})')

if __name__ == '__main__':
    main()