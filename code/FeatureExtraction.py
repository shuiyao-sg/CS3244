import pandas as pd
import numpy.polynomial.polynomial as poly
from datetime import datetime

WINDOW_SIZE_MINUTE = 150
STEP_SIZE_MINUTE = 15
WINDOW_SIZE_INT = 10
DURATION_IN_SECONDS = 8100.0
STEP_SIZE_INT = 1
X_AXIS_ARRAY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def extract_sensor_input(df, x_axis_array, output_columns):
    column_headers = df.columns
    output_data = []

    for row in range(0, len(df) - WINDOW_SIZE_INT + 1, STEP_SIZE_INT):
        temp_tuple = []
        end_index = row + WINDOW_SIZE_INT - 1

        start_datetime = datetime.strptime(df[column_headers[1]][row], "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(df[column_headers[1]][end_index], "%Y-%m-%d %H:%M:%S")
        time_delta = end_datetime - start_datetime
        if time_delta.total_seconds() != DURATION_IN_SECONDS:
            continue

        for col in range(1, 4):
            temp_tuple.append(df[column_headers[col]][end_index])
        temp_tuple.append(df[column_headers[7]][end_index])  # append UNIX TIME at end_index row

        for col in range(4, 7):
            col_data_array = df[column_headers[col]][row:(end_index + 1)]
            mean_val = col_data_array.mean()
            min_val = col_data_array.min()
            max_val = col_data_array.max()
            # lrc stands for linear regression coefficient
            lrc_result_array = poly.polyfit(x=x_axis_array, y=col_data_array, deg=1)
            lrc = lrc_result_array[1]
            temp_tuple.append(mean_val)
            temp_tuple.append(min_val)
            temp_tuple.append(max_val)
            temp_tuple.append(lrc)
        output_data.append(temp_tuple)
    return pd.DataFrame(output_data, columns=output_columns)


if __name__ == "__main__":
    input_files = ["../data/Extract/plant_1_collate_raw_filled.csv", "../data/Extract/plant_2_collate_raw_filled.csv"]
    sensor_input_collate_files = ["../data/Extract/plant_1_sensor_input_collate.csv",
                                  "../data/Extract/plant_2_sensor_input_collate.csv"]
    sensor_input_collate_column_headers = ["date_time", "plant_id", "source_key", "unix time",
                                           "mean ambient temperature", "min ambient temperature", "max ambient temperature",
                                           "lrc ambient temperature",
                                           "mean module temperature", "min module temperature", "max module temperature",
                                           "lrc module temperature",
                                           "mean irradiation", "min irradiation", "max irradiation",
                                           "lrc irradiation"]  # "lrc" stands for linear regression coefficient
    for i in range(0, 2):
        data_frame = pd.read_csv(input_files[i])
        output_df = extract_sensor_input(data_frame, X_AXIS_ARRAY, sensor_input_collate_column_headers)
        output_df.to_csv(sensor_input_collate_files[i])
