import torch

def balance_data(data, label):
    """
    过采样
    :param data: data
    :param label: label
    :return:
    """
    # 找到正类和负类的索引
    positive_indices = torch.where(label == 1)[0]
    negative_indices = torch.where(label == 0)[0]

    # 计算正类和负类的数量
    num_positive = len(positive_indices)
    num_negative = len(negative_indices)

    # 过采样，使正类=负类
    if num_positive < num_negative:
        # 对正类进行过采样
        oversampled_indices = torch.randint(num_positive, (num_negative - num_positive,))
        positive_indices = torch.cat((positive_indices, positive_indices[oversampled_indices]))
    elif num_positive > num_negative:
        # 对负类进行过采样
        oversampled_indices = torch.randint(num_negative, (num_positive - num_negative,))
        negative_indices = torch.cat((negative_indices, negative_indices[oversampled_indices]))

    # 确保正类和负类数量相等后，重新排列数据并打乱顺序
    indices = torch.cat((positive_indices, negative_indices))
    shuffled_indices = torch.randperm(len(indices))
    indices = indices[shuffled_indices]
    data_balanced = data[indices]
    label_balanced = label[indices]

    return data_balanced, label_balanced