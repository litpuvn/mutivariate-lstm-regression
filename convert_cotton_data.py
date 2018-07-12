from pandas import read_csv
from datetime import datetime
import pandas as pd

label_column = 'CLS-AVGYLD'
label_column = 'AVGYLD'
data_columns = ['ELE', 'SLOPE', 'CURV', 'PRO', 'PLAN', 'EC_SH', 'EC_DP', 'BAND1', 'BAND2', 'BAND3', 'BAND4', 'VI_AVG']

df = pd.read_csv('data/data_excel_converted.csv')
selected = [label_column] + data_columns
non_selected = list(set(df.columns) - set(selected))

df = df.drop(non_selected, axis=1) # Drop non selected columns
df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows

df.to_csv('data/cotton.csv', index=False)