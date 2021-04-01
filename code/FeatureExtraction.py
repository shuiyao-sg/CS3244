import pandas as pd
import numpy.polynomial.polynomial as poly

WINDOW_SIZE_MINUTE = 150
STEP_SIZE_MINUTE = 15
WINDOW_SIZE_INT = 10
STEP_SIZE_INT = 1
X_AXIS_ARRAY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

df = pd.read_csv("../data/plant_1_collate_raw.csv")

OUTPUT_COLUMNS = ["date_time", "plant_id", "source_key", "unix time",
                  "mean ambient temperature", "min ambient temperature", "max ambient temperature",
                  "lrc ambient temperature",
                  "mean module temperature", "min module temperature", "max module temperature",
                  "lrc module temperature",
                  "mean irradiation", "min irradiation", "max irradiation",
                  "lrc irradiation"]  # "lrc" stands for linear regression coefficient


def extract_features(df, x_axis_array, output_columns):
    column_headers = df.columns
    output_data = []
    for row in range(0, len(df) - WINDOW_SIZE_INT + 1, STEP_SIZE_INT):
        temp_tuple = []
        end_index = row + WINDOW_SIZE_INT - 1
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
    input_files = ["../data/plant_1_collate_raw.csv", "../data/plant_2_collate_raw.csv"]
    output_files = ["../data/plant_1_preprocessing.csv", "../data/plant_2_preprocessing.csv"]
    for i in range(0, 2):
        data_frame = pd.read_csv(input_files[i])
        output_df = extract_features(data_frame, X_AXIS_ARRAY, OUTPUT_COLUMNS)
        output_df.to_csv(output_files[i])
