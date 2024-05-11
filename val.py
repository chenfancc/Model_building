import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def calculate_metrics(true_labels_flat, predicted_probs_flat, best_threshold):
    # 计算混淆矩阵
    TP = np.sum((predicted_probs_flat > best_threshold) & (true_labels_flat == 1))
    FP = np.sum((predicted_probs_flat > best_threshold) & (true_labels_flat == 0))
    TN = np.sum((predicted_probs_flat <= best_threshold) & (true_labels_flat == 0))
    FN = np.sum((predicted_probs_flat <= best_threshold) & (true_labels_flat == 1))

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    # 计算特异性、敏感性和准确率
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    alarm_accuracy = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return confusion_matrix, specificity, sensitivity, alarm_accuracy, accuracy


def plot_confusion_matrix(cm, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 设置其他字体属性，如字号
    plt.rcParams.update({'font.size': 12})
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("归一化混淆矩阵")
    else:
        print('混淆矩阵，未归一化')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for idx in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, idx, format(cm[idx, j], fmt),
                     horizontalalignment="center",
                     color="red" if cm[idx, j] > thresh else "red")

    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.tight_layout()
    plt.show()


def validation(data_loader, model):
    model.eval()  # 设置为评估模式
    true_labels = []
    predicted_probs = []
    # 对测试数据进行预测
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.float()
            targets = targets.long()
            outputs = model(data).float()
            true_labels.append(targets.numpy())
            predicted_probs.append(outputs.numpy())

    true_labels_flat = np.concatenate(true_labels)
    predicted_probs_flat = np.concatenate(predicted_probs)

    fpr, tpr, thresholds = roc_curve(true_labels_flat, predicted_probs_flat)
    roc_auc = auc(fpr, tpr)
    best_threshold_index = (10 * tpr - fpr).argmax()
    best_threshold = thresholds[best_threshold_index]

    print(f"AUC: {roc_auc:.2f}")
    print(f"Best threshold: {best_threshold:.2f}")

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    confusion_matrix, specificity, sensitivity, alarm_accuracy, accuracy = calculate_metrics(true_labels_flat,
                                                                                             predicted_probs_flat,
                                                                                             best_threshold)

    print("Confusion Matrix:")
    print(confusion_matrix)
    print(f"Specificity: {specificity:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Alarm Accuracy: {alarm_accuracy:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(confusion_matrix, classes=['Survive', 'Death'])

    return confusion_matrix, specificity, sensitivity, alarm_accuracy, accuracy
