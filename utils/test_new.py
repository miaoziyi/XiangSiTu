# a = 'img'
# img_json = a.replace('i', '')
# print(a)

# a = 'img_test/new/26_11_980.jpg'.replace('img', '').replace('26', '')
# print(a)

import numpy as np
import pandas as pd

# image_ids = pd.read_csv('txt/per_leafCid_250_StratifiedKFold.csv')
# print(len(image_ids['spuId'].unique()))

# error = [1, 3]
# np.load('error', error)
# error = np.load('error_10000.npy')
# print(len(error))

import pandas as pd
dict_data = {
	'student':["Li Lei","Han Meimei","Tom"],
	'score'	:[95,98,92],
	'gender':['M','F','M']
}
DF_data = pd.DataFrame(dict_data,columns=['gender','student','score'])

print(DF_data)
print(DF_data.loc[2])
DF_data=DF_data.drop(1)
print(DF_data)
# DF_data=DF_data.drop(2)
# print(DF_data)
# DF_data.to_csv('test.csv', index=False)
print(DF_data.loc[2])
