import pandas as pd
from extract_func.Extract_PTE_function import *

df = pd.read_csv('GPS_stations/product2/UNRcombinedGPS_ztd.csv')
df = df[df['sigZTD'] < 0.1]

extract_wethydro_GNSS(df, 'weather_model/weather_files/', fixed_hgt=True)
