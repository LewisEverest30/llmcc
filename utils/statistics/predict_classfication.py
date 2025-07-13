# 初始化
num_classes = 3
true_labels = []
pred_labels = []

# 读取文件
with open('output\\test\\predict_result_2025-07-10-21-29-12.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        true_labels.append(int(parts[0]))
        pred_labels.append(int(parts[1]))

# 初始化统计
TP = [0] * num_classes  # True Positive  真实是 k 且预测也是 k 的个数
FP = [0] * num_classes  # False Positive  预测是 k，但真实不是 k 的个数
FN = [0] * num_classes  # False Negative  真实是 k，但预测不是 k 的个数
support = [0] * num_classes  # 每类真实数量

# 遍历统计
for t, p in zip(true_labels, pred_labels):
    support[t] += 1
    if t == p:
        TP[t] += 1
    else:
        FP[p] += 1
        FN[t] += 1

# 计算指标
for k in range(num_classes):
    precision = TP[k] / (TP[k] + FP[k]) if (TP[k] + FP[k]) > 0 else 0.0
    recall = TP[k] / (TP[k] + FN[k]) if (TP[k] + FN[k]) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Class {k}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    # print(f"  F1 score:  {f1:.4f}")
    # print(f"  Support:   {support[k]}")

# 总体准确率
total_correct = sum(TP)
total = len(true_labels)
accuracy = total_correct / total if total > 0 else 0.0
print(f"\nOverall Accuracy: {accuracy:.4f}")
