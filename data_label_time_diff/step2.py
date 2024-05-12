import pandas as pd
import numpy as np
from tqdm import tqdm


def tick_label(numpy_array, time_DIF, is_blood=None):
    time_column = np.zeros(numpy_array.shape[0], dtype=float)
    for stay_id in tqdm(np.unique(numpy_array[:, 2]), desc="Processing IDs"):  # 添加进度条
        # print(id)
        first_row_index = np.where(numpy_array[:, 2] == stay_id)[0][0]
        # print(first_row_index)
        first_time = pd.Timestamp(numpy_array[first_row_index, 5])
        # print(first_time)

        rows_with_id = np.where(numpy_array[:, 2] == stay_id)[0]
        # print(rows_with_id)
        time_column[rows_with_id] = (
                                            pd.to_datetime(
                                                numpy_array[rows_with_id, 5]) - first_time).total_seconds() // 3600
        # print(time_column[rows_with_id])
        # break
    numpy_array = np.column_stack((numpy_array, time_column))

    label_column = np.zeros(numpy_array.shape[0], dtype=int)
    for idx in tqdm(range(numpy_array.shape[0]), desc="Processing rows"):  # 添加进度条
        time_diff = (pd.to_datetime(numpy_array[idx, 3]) -
                     pd.to_datetime(numpy_array[idx, 5])).total_seconds() // 3600
        if time_diff <= time_DIF:
            label_column[idx] = 1
        else:
            label_column[idx] = 0
        # break
    numpy_array = np.column_stack((numpy_array, label_column))

    numpy_array_dropped = np.delete(numpy_array, [3, 4, 5], axis=1)

    if not is_blood:
        df = pd.DataFrame(numpy_array_dropped, columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
        df_mean = df.groupby(['col1', 'col2', 'col3', 'col5']).mean().reset_index()
    else:
        df = pd.DataFrame(numpy_array_dropped, columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'])
        df_mean = df.groupby(['col1', 'col2', 'col3', 'col7']).mean().reset_index()
    return df_mean


df_1 = pd.read_excel("blood_pressure_with_label.xlsx")
numpy_array_1 = df_1.to_numpy()
df_2 = pd.read_excel("heart_rate_with_label.xlsx")
numpy_array_2 = df_2.to_numpy()
df_3 = pd.read_excel("respiratory_rate_with_label.xlsx")
numpy_array_3 = df_3.to_numpy()
df_4 = pd.read_excel("spo2_with_label.xlsx")
numpy_array_4 = df_4.to_numpy()
df_5 = pd.read_excel("temperature_with_label.xlsx")
numpy_array_5 = df_5.to_numpy()

for i in [4, 6, 8, 12, 24]:
    data_1 = tick_label(numpy_array_1, i, True)
    data_2 = tick_label(numpy_array_2, i)
    data_3 = tick_label(numpy_array_3, i)
    data_4 = tick_label(numpy_array_4, i)
    data_5 = tick_label(numpy_array_5, i)

    np.savetxt(f"data_1—{i}.csv", data_1, delimiter=",", fmt="%s")
    np.savetxt(f"data_2—{i}.csv", data_2, delimiter=",", fmt="%s")
    np.savetxt(f"data_3—{i}.csv", data_3, delimiter=",", fmt="%s")
    np.savetxt(f"data_4—{i}.csv", data_4, delimiter=",", fmt="%s")
    np.savetxt(f"data_5—{i}.csv", data_5, delimiter=",", fmt="%s")
