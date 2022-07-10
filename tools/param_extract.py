from extract_func.Extract_PTE_function import *
import pandas as pd

hgtlvs = [-100, 0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400
    , 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000
    , 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000
    , 14000, 15000, 16000, 17000, 18000, 19000, 20000, 25000, 30000, 35000, 40000]
print('US_west')
df = pd.read_csv('US_west/UNRcombinedGPS_ztd.csv')
df = df[df['sigZTD'] < 1].reset_index(drop=True)
print(len(df))
extract_param_GNSS(df, wm_file_path='../GNSS/US_west/weather_files/', workLoc='US_west/')

print('US_east')
df = pd.read_csv('US_east/UNRcombinedGPS_ztd.csv')
df = df[df['sigZTD'] < 1].reset_index(drop=True)
print(len(df))
extract_param_GNSS(df, wm_file_path='../GNSS/US_east/weather_files/', workLoc='US_east/')
