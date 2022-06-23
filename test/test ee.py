import ee
from tools.Extract_ee_function import *
service_account = 'goes-extract@extract-goes-1655507865824.iam.gserviceaccount.com'
KEY = 'tools/private_key.json'
credentials = ee.ServiceAccountCredentials(service_account, KEY)
ee.Initialize(credentials)

# create AOI
area = addGeometry(-89, -88, 36, 40)

test_goes = get_GOES16_image('2017-01-01T11:00:00', area)

# Scale and offset the image
Img = applyScaleAndOffset_all(test_goes)

print(test_goes.getInfo())
# from google.auth.transport.requests import AuthorizedSession
# from google.oauth2 import service_account
#
# credentials = service_account.Credentials.from_service_account_file(KEY)
# scoped_credentials = credentials.with_scopes(
#     ['https://www.googleapis.com/auth/cloud-platform'])
#
# session = AuthorizedSession(scoped_credentials)
#
# url = 'https://earthengine.googleapis.com/v1alpha/projects/earthengine-public/assets/LANDSAT'
#
# response = session.get(url)
#
# from pprint import pprint
# import json
#
# pprint(json.loads(response.content))
