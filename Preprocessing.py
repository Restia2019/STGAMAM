import pandas as pd
import time
import numpy as np
import datetime
from icecream import ic


def pre_process(source, new_source):
    df = pd.read_csv(source)

    time_column = df.columns[0]
    data_columns = df.columns[1:]

    new_df = pd.DataFrame()
    index = 1
    for col in data_columns:
        temp_df = pd.DataFrame()
        temp_df[time_column] = df[time_column]
        temp_df['station_id'] = index
        temp_df['humidity'] = df[col]
        temp_df['reindexed_id'] = index
        new_df = pd.concat([new_df, temp_df])
        index = index + 1

    new_df.reset_index(drop=True, inplace=True)

    new_df.to_csv(new_source, index=False)

    df = pd.read_csv(new_source)

    column1 = 'datetime'
    column2 = 'station_id'

    columns = df.columns.tolist()
    index1 = columns.index(column1)
    index2 = columns.index(column2)

    columns[index1], columns[index2] = columns[index2], columns[index1]

    df = df[columns]

    df.to_csv(new_source, index=False)


def process_data(source):
    df = pd.read_csv(source)

    timestamps = [ts.split('+')[0] for ts in df['datetime']]
    timestamps_hour = np.array([float(datetime.datetime.strptime(t, '%Y/%m/%d %H:%M').hour) for t in timestamps])
    timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y/%m/%d %H:%M').day) for t in timestamps])
    timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y/%m/%d %H:%M').month) for t in timestamps])

    hours_in_day = 24
    days_in_month = 30
    month_in_year = 12

    df['sin_hour'] = np.sin(2 * np.pi * timestamps_hour / hours_in_day)
    df['cos_hour'] = np.cos(2 * np.pi * timestamps_hour / hours_in_day)
    df['sin_day'] = np.sin(2 * np.pi * timestamps_day / days_in_month)
    df['cos_day'] = np.cos(2 * np.pi * timestamps_day / days_in_month)
    df['sin_month'] = np.sin(2 * np.pi * timestamps_month / month_in_year)
    df['cos_month'] = np.cos(2 * np.pi * timestamps_month / month_in_year)

    return df


pre_process('Data/humidity_train_raw.csv', 'Data/humidity_train.csv')
pre_process('Data/humidity_test_raw.csv', 'Data/humidity_test.csv')

train_dataset = process_data('Data/humidity_train.csv')
train_dataset.to_csv(r'Data/humidity_train_data.csv', index=False)

test_dataset = process_data('Data/humidity_test.csv')
test_dataset.to_csv(r'Data/humidity_test_data.csv', index=False)
