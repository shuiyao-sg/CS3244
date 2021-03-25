import pandas as pd
import matplotlib.pyplot as plt

WINDOW_SIZE_MINUTE = 150
STEP_SIZE_MINUTE = 15
WINDOW_SIZE_INT = 10
STEP_SIZE_INT = 1

df = pd.read_csv("../data/plant_1_collate_raw.csv")
column_headers = df.columns
output_columns = ["date_time", "plant_id", "source_key", "unix time",
                  "mean ambient temperature", "min ambient temperature", "max ambient temperature",
                  "mean module temperature", "min module temperature", "max module temperature",
                  "mean irradiation", "min irradiation", "max irradiation"]
output_data = []

for row in range(0, len(df) - WINDOW_SIZE_INT + 1, STEP_SIZE_INT):
    temp_tuple = []
    col = 0
    end_index = row + WINDOW_SIZE_INT - 1
    for col in range(1, 4):
        temp_tuple.append(df[column_headers[col]][end_index])
    temp_tuple.append(df[column_headers[7]][end_index])  # append UNIX TIME at end_index row

    for col in range(4, 7):
        col_data_array = df[column_headers[col]][row:(end_index + 1)]
        mean_val = col_data_array.mean()
        min_val = col_data_array.min()
        max_val = col_data_array.max()
        temp_tuple.append(mean_val)
        temp_tuple.append(min_val)
        temp_tuple.append(max_val)
    output_data.append(temp_tuple)

output_df = pd.DataFrame(output_data, columns=output_columns)
output_df.to_csv("../data/mean_min_max_sensor_data.csv")

# for plotting scatter diagram
# output_df.plot.scatter(x="date_time", y="mean ambient temperature")
# plt.savefig(fname="figure1.png", dpi=300)

print("end")
