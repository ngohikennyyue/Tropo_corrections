import pandas as pd
from extract_func.Extract_closest4_function import *

df = pd.read_csv('GPS_stations/product2/UNRcombinedGPS_ztd.csv')

extract_param_GNSS_4Closest(df, 'weather_model/weather_files/',fixed_hgt=True)