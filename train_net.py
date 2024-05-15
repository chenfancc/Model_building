import os
from datetime import datetime

import torch.cuda
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from val import validation


def train_val_net_old(model_name, epoch, model, train_dataloader, val_dataloader, loss_fn, optimizer, threshold=0.5):
    writer = SummaryWriter(f"logs_train_of_{model_name}")
    total_train_step = 0
    train_loss_list = []
    val_loss_list = []
    accuracy_list = []
    alarm_sen_list = []
    alarm_acc_list = []

    for i in range(epoch):
        assert epoch is not None and epoch > 0, "EPOCH错误"
        print("-------第 {} 轮训练开始-------".format(i + 1))
        model.train()
        for item in train_dataloader:
            data, targets = item
            data = data.float()
            targets = targets.long()
            if torch.cuda.is_available():
                data = data.cuda()
                targets = targets.cuda()
            outputs = model(data)
            loss = loss_fn(outputs, targets.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}, Device:{}".format(total_train_step, loss.item(), loss.device))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        train_loss_list.append(loss.item())
        model.eval()    
        eps = 1e-6
        total_val_loss = eps
        total_accuracy = eps
        total_num = eps
        total_TP = eps
        total_TN = eps
        total_FP = eps
        total_FN = eps
        count = 0
        with torch.no_grad():
            for item in val_dataloader:
                count += 1
                data, targets = item
                data = data.float()
                targets = targets.long()
                if torch.cuda.is_available():
                    data = data.cuda()
                    targets = targets.cuda()
                outputs = model(data)
                # print(outputs.shape)
                outputs_bi = (outputs >= threshold).float()
                # print(outputs_bi)
                loss = loss_fn(outputs, targets.float())
                total_val_loss = total_val_loss + loss.item()
                TP = ((outputs_bi == 1) & (targets == 1)).sum().item()  # 预测为正类且实际为正类的数量
                TN = ((outputs_bi == 0) & (targets == 0)).sum().item()  # 预测为负类且实际为负类的数量
                FP = ((outputs_bi == 1) & (targets == 0)).sum().item()  # 预测为正类但实际为负类的数量
                FN = ((outputs_bi == 0) & (targets == 1)).sum().item()  # 预测为负类但实际为正类的数量
                accuracy = TP + TN
                total_accuracy = total_accuracy + accuracy
                test_num = TP + TN + FP + FN
                total_num += test_num
                total_TP = total_TP + TP
                total_TN = total_TN + TN
                total_FP = total_FP + FP
                total_FN = total_FN + FN
            print("整体测试集上的Loss: {}".format(total_val_loss / count))
            val_loss_list.append(total_val_loss / count)
            print("整体测试集上的正确率: {}".format(total_accuracy / total_num))
            accuracy_list.append(total_accuracy / total_num)
            print("整体测试集上的报警灵敏度: {}".format(total_TP / (total_TP + total_FN)))
            alarm_sen_list.append(total_TP / (total_TP + total_FN))
            print("整体测试集上的报警准确度: {}".format(total_TP / (total_TP + total_FP)))
            alarm_acc_list.append(total_TP / (total_TP + total_FP))
            writer.add_scalar("val_loss", total_val_loss, i + 1)
            writer.add_scalar("val_accuracy", total_accuracy / total_num, i + 1)
            model_directory = f"./{model_name}/"
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            torch.save(model, f"{model_name}/{model_name}_{i}.pth")
            print("模型已保存")
        writer.close()
    info = {
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "accuracy_list": accuracy_list,
        "alarm_sen_list": alarm_sen_list,
        "alarm_acc_list": alarm_acc_list
    }
    return info

def train_val_net(model_name, epoch, model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler):
    total_train_step = 0
    train_loss_list = []
    train_loss_total_list = []
    val_loss_list = []
    accuracy_list = []
    specificity_list = []
    alarm_sen_list = []
    alarm_acc_list = []

    for i in range(epoch):
        assert epoch is not None and epoch > 0, "EPOCH错误"
        print("-------第 {} 轮训练开始-------".format(i + 1))
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间
        print("开始时间", current_time)
        model.train()
        model.to("cuda")
        for item in train_dataloader:
            data, targets = item
            data = data.float()
            targets = targets.long()
            if torch.cuda.is_available():
                data = data.cuda()
                targets = targets.cuda()
            outputs = model(data)
            loss = loss_fn(outputs, targets.float())
            train_loss_total_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}, Device:{}".format(total_train_step, loss.item(), loss.device))
        scheduler.step()
        train_loss_list.append(loss.item())
        model.eval()
        eps = 1e-6
        total_val_loss = eps
        true_labels = []
        count = 0
        with torch.no_grad():
            for item in val_dataloader:
                count += 1
                data, targets = item
                data = data.float()
                targets = targets.long()
                if torch.cuda.is_available():
                    data = data.cuda()
                    targets = targets.cuda()
                outputs = model(data)
                true_labels.append(targets.cpu().numpy())
                loss = loss_fn(outputs, targets.float())
                total_val_loss = total_val_loss + loss.item()
            print("整体测试集上的Loss: {}".format(total_val_loss / count))
            val_loss_list.append(total_val_loss / count)
            model_directory = f"./{model_name}/"
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            _, specificity, sensitivity, alarm_accuracy, accuracy = validation(val_dataloader, model.to("cpu"), model_name, i)
            accuracy_list.append(accuracy)
            specificity_list.append(specificity)
            alarm_sen_list.append(sensitivity)
            alarm_acc_list.append(alarm_accuracy)
            torch.save(model, f"{model_name}/{model_name}_{i}.pth")
            print("模型已保存")
    info = {
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
        "accuracy_list": accuracy_list,
        "specificity_list": specificity_list,
        "alarm_sen_list": alarm_sen_list,
        "alarm_acc_list": alarm_acc_list,
        "train_loss_total_list": train_loss_total_list
    }
    return info

import torch.cuda
from torch.utils.tensorboard import SummaryWriter


def train_net(model_name, epoch_total, model, train_dataloader, val_dataloader, loss_fn, optimizer, device):
    for epoch in range(epoch_total):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        for data, label in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epoch_total}'):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()  # 梯度清零

            # 正向传播
            outputs = model(data)
            # print(outputs.shape, label.shape)
            loss = loss_fn(outputs, label)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        # 计算平均损失
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f'Epoch [{epoch + 1}/{epoch_total}], Loss: {epoch_loss:.4f}')
    torch.save(model, f"{model_name}.pth")
