from Extract_ee_function import *

service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
KEY = 'private_key.json'
credentials = ee.ServiceAccountCredentials(service_account, KEY)

Num_bands = 32
print('Initialize Google Earth Engine...')
ee.Initialize(credentials)
print('Reading in file...')
extract_param('PTE_vert_fixed_hgtlvs.csv', '11:00:00', ['CMI_C08', 'CMI_C09', 'CMI_C10', 'CMI_C12'])
print('Finished extraction')
# data.to_csv('US_PTE_cloud.csv', index=False)
