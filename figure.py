import json
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_figure(info, model_name):
    file_path = os.path.join(f"./{model_name}", "data_label_1.json")
    with open(file_path, "w") as file:
        json.dump(info, file)
    train_loss_list = info["train_loss_list"]
    val_loss_list = info["val_loss_list"]
    accuracy_list = info["accuracy_list"]
    specificity_list = info["specificity_list"]
    alarm_sen_list = info["alarm_sen_list"]
    alarm_acc_list = info["alarm_acc_list"]
    train_loss_total_list = info["train_loss_total_list"]
    
    accuracy_tensor = torch.tensor(accuracy_list)  # Convert list to PyTorch tensor
    specificity_tensor = torch.tensor(specificity_list)  # Convert list to PyTorch tensor
    alarm_sen_tensor = torch.tensor(alarm_sen_list)  # Convert list to PyTorch tensor
    alarm_acc_tensor = torch.tensor(alarm_acc_list)  # Convert list to PyTorch tensor
    accuracy_list_cpu = accuracy_tensor.cpu()  # Move tensor to CPU memory
    specificity_list_cpu = specificity_tensor.cpu()  # Move tensor to CPU memory
    alarm_sen_cpu = alarm_sen_tensor.cpu()
    alarm_acc_cpu = alarm_acc_tensor.cpu()

    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.savefig(f"{model_name}/{model_name}_loss.png")
    plt.show()

    plt.clf()

    plt.plot(accuracy_list_cpu, label='Accuracy')
    plt.plot(specificity_list_cpu, label='specificity')
    plt.plot(alarm_sen_cpu, label='Alarm Sensitivity')
    plt.plot(alarm_acc_cpu, label='Alarm Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()

    plt.savefig(f"{model_name}/{model_name}_info.png")
    plt.show()

    # 创建x轴数据，假设为等间距的点
    x = np.arange(len(train_loss_total_list))

    # 绘制曲线
    plt.plot(x, train_loss_total_list)

    # 添加标题和标签
    plt.title('Label Curve')
    plt.xlabel('Index')
    plt.ylabel('Label Value')

    # 显示网格
    plt.grid(True)
    plt.savefig(f"{model_name}/{model_name}_train_loss.png")
    plt.show()
