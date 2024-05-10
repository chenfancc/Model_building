import json
import os
import pickle

import torch
from matplotlib import pyplot as plt


def plot_figure(info, model_name):
    file_path = os.path.join(f"./{model_name}", "data.json")
    with open(file_path, "w") as file:
        json.dump(info, file)
    train_loss_list = info["train_loss_list"]
    val_loss_list = info["val_loss_list"]
    accuracy_list = info["accuracy_list"]
    alarm_sen_list = info["alarm_sen_list"]
    alarm_acc_list = info["alarm_acc_list"]
    accuracy_tensor = torch.tensor(accuracy_list)  # Convert list to PyTorch tensor
    alarm_sen_tensor = torch.tensor(alarm_sen_list)  # Convert list to PyTorch tensor
    alarm_acc_tensor = torch.tensor(alarm_acc_list)  # Convert list to PyTorch tensor
    accuracy_list_cpu = accuracy_tensor.cpu()  # Move tensor to CPU memory
    alarm_sen_cpu = alarm_sen_tensor.cpu()
    alarm_acc_cpu = alarm_acc_tensor.cpu()

    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()

    plt.show()
    plt.savefig(f"{model_name}/{model_name}_1.png")
    plt.clf()

    plt.plot(accuracy_list_cpu, label='Accuracy')
    plt.plot(alarm_sen_cpu, label='Alarm Sensitivity')
    plt.plot(alarm_acc_cpu, label='Alarm Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()

    plt.show()
    plt.savefig(f"{model_name}/{model_name}_2.png")
