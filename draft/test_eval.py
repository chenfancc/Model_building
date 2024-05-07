import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from draft.simulation_data import synthetic_data

if __name__ == '__main__':
    for i in range(20):
        beta = i/10
        data_test, label_test = synthetic_data(10, 24, 7, beta)
        dataset_test = TensorDataset(data_test, label_test)
        test_dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True)

        # 加载整个模型对象
        model = torch.load("test_bilstm.pth")
        model.to("cpu")

        model.eval()
        i = 0
        accuracy_total = 0
        for input_data, label_data in test_dataloader:
            # input_data, label_data = input_data.to("cuda"), label_data.to("cuda")
            # 进行预测
            with torch.no_grad():
                outputs = model(input_data)
            y_1_values = data_test[i, :, 0].tolist()
            x_values = torch.arange(24).tolist()
            if i < 5:
                plt.plot(x_values, y_1_values, color="red")
            else:
                plt.plot(x_values, y_1_values, color="green")
            plt.ylim(-2, 1)

            # print(outputs)
            # print(label_data)
            eps = 1e-2
            predicted = (outputs.item() - label_data.item() < eps) and (outputs.item() - label_data.item() > -eps)
            accuracy_total += int(predicted)
            i += 1
        print(f"{beta}_Accuracy:", accuracy_total/label_test.shape[0])
        plt.show()
        plt.clf()
