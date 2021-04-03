import pandas as pd
import numpy.polynomial.polynomial as poly
from datetime import datetime

WINDOW_SIZE_MINUTE = 150
STEP_SIZE_MINUTE = 15
WINDOW_SIZE_INT = 10
DURATION_IN_SECONDS = 8100.0
STEP_SIZE_INT = 1
X_AXIS_ARRAY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def extract_features(df, x_axis_array, output_columns):
    column_headers = df.columns
    dc_headers = column_headers[30:52]
    output_data = []

    for row in range(0, len(df) - WINDOW_SIZE_INT + 1, STEP_SIZE_INT):
        temp_tuple = []
        end_index = row + WINDOW_SIZE_INT - 1

        if not is_valid_window(df, column_headers, row, end_index):
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
        extract_dc_output(df, dc_headers, end_index, temp_tuple)
        output_data.append(temp_tuple)
    return pd.DataFrame(output_data, columns=output_columns)


def extract_dc_output(df, input_column_headers, end_index, output_tuple):
    for col in range(0, 22):
        for row in range(1, 5):
            dc_forcast = df[input_column_headers[col]][end_index + row]
            output_tuple.append(dc_forcast)


def is_valid_window(df, column_headers, start_index, end_index):
    start_datetime = datetime.strptime(df[column_headers[1]][start_index], "%Y-%m-%d %H:%M:%S")
    is_valid = True
    offset_in_second = 0.0
    for i in range(0, 5):
        if end_index + i >= len(df):
            is_valid = False
            return is_valid
        end_datetime = datetime.strptime(df[column_headers[1]][end_index + i], "%Y-%m-%d %H:%M:%S")
        time_delta = end_datetime - start_datetime
        if time_delta.total_seconds() != DURATION_IN_SECONDS + offset_in_second:
            # debug
            print("start = " + str(start_datetime) + " i = " + str(i) + " i-time = " + str(end_datetime))
            # end debug
            is_valid = False
            return is_valid
        offset_in_second += 900.0  # 15 min
    return is_valid


if __name__ == "__main__":
    input_files = ["../data/Extract/plant_1_collate_raw_filled.csv", "../data/Extract/plant_2_collate_raw_filled.csv"]
    output_files = ["../data/Extract/plant_1_feature_extract.csv",
                    "../data/Extract/plant_2_feature_extract.csv"]
    output_column_headers = ["date_time", "plant_id", "source_key", "unix time",
                             "mean ambient temperature", "min ambient temperature", "max ambient temperature",
                             "lrc ambient temperature",
                             "mean module temperature", "min module temperature", "max module temperature",
                             "lrc module temperature",
                             "mean irradiation", "min irradiation", "max irradiation",
                             "lrc irradiation"]  # "lrc" stands for linear regression coefficient

    # adding dc column headers
    durations_in_min = [15, 30, 45, 60]
    for col in range(0, 22):
        for row in range(0, 4):
            header = "DC_" + str(col) + "_post_" + str(durations_in_min[row]) + "_min_forcast"
            output_column_headers.append(header)

    for i in range(0, 2):
        data_frame = pd.read_csv(input_files[i])
        data_extract = extract_features(data_frame, X_AXIS_ARRAY, output_column_headers)
        data_extract.to_csv(output_files[i])
