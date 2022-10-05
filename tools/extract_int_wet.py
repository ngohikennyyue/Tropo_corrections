import sys
import os
from extract_func.Extract_PTE_function import *
from extract_func.Extract_int_wm_wet import *

df = pd.read_csv('GPS_stations/product2/UNRcombinedGPS_ztd.csv')
df = df.dropna()

extract_inter_param(df, wm_file_path='weather_model/weather_files/')

print('Finished extraction')


fig, ax = plt.subplots()
ax.scatter(LSU1.start_date, LSU1.hydro1, color='red', label='hydro')
ax.tick_params(axis='y', labelcolor='red')
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax2 = ax.twinx()
ax2.scatter(LSU1.start_date,LSU1.wet1, color='green', label='wet')
ax2.tick_params(axis='y', labelcolor='green')
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
fig.legend()
fig.suptitle('WM_Hydro_wet_GPSstation_1LSU')
plt.savefig('1LSU_WM_hydro_delay.png')