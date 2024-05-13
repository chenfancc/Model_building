import os

import pandas as pd
from tqdm import tqdm


def fill_missing_rows(data):
    filled_data = []
    prev_id = None
    prev_time = None
    prev_label = None
    total_count = 0
    for row in tqdm(data):
        id_, time, *other_cols, label = row

        if prev_id is None:
            prev_id = id_
            prev_time = time
            prev_label = label

        count = 0
        while prev_id == id_ and prev_time < time:
            new_row = [prev_id, prev_time] + [0] * (len(other_cols)) + [0]
            if prev_label == 1 and label == 1:
                new_row[-1] = 1
            filled_data.append(new_row)
            # print(new_row)
            count += 1
            prev_time += 1

        if count > 10: total_count += 1
        filled_data.append(row)

        prev_id = id_
        prev_time = time + 1
        prev_label = label

    print(total_count)

    return filled_data


def run_step3_label_1(not_test=True):
    if not_test:
        data_dirs = ["../data", "../data/label_1", "../data/label_1/filled_data"]  # 定义需要检查的目录列表

        for directory in data_dirs:  # 检查并创建目录
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory '{directory}' created.")

        for i in [1, 2, 3, 4, 5]:
            for j in [20, 24, 30, 36, 48]:
                data = pd.read_csv(f"../data/label_1/ticked_data/data_{i}_{j}.csv")
                first_row = data.iloc[0].values

                data_np = data.to_numpy()

                result = fill_missing_rows(data_np)
                if i == 1:
                    column = ['stay_id', 'sbp', 'dbp', 'map', 'time', 'label']
                elif i == 2:
                    column = ['stay_id', 'hr', 'time', 'label']
                elif i == 3:
                    column = ['stay_id', 'rr', 'time', 'label']
                elif i == 4:
                    column = ['stay_id', 'spo2', 'time', 'label']
                elif i == 5:
                    column = ['stay_id', 'temp', 'time', 'label']
                result_df = pd.DataFrame(result, columns=column)
                result_df.to_csv(f"../data/label_1/filled_data/filled_data_{i}_{j}.csv", index=False)

                print(result_df[:3])
    else:
        data = pd.read_csv(f"../data/label_1/ticked_data/test.csv", header=None)
        first_row = data.iloc[0].values

        data_np = data.to_numpy()

        result = fill_missing_rows(data_np)
        column_names = ['stay_id', 'sbp', 'dbp', 'map', 'time', 'label']
        result_df = pd.DataFrame(result, columns=column_names)
        # result_df[['stay_id', 'time', 'label']] = result_df[['stay_id', 'time', 'label']].astype(int)

        # results = [first_row] + result
        result_df.to_csv(f"../data/label_1/filled_data/test.csv", index=False)

        print(result_df[:3])
