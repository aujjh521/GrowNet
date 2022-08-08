'''
前處理ovl data, save to .npz
'''
#general
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#load data
print(f'{"="*20} start load data {"="*20}')
train_data_path = 'data/OVL_data_for_ML_test_medium_new.xlsx'
test_data_path = 'data/OVL_data_for_ML_test_medium_exam_new.xlsx'

train_dataset = pd.read_excel(train_data_path)
test_dataset = pd.read_excel(test_data_path)
print(f'train / test dataset are loaded, shape is {train_dataset.shape} / {test_dataset.shape}')

#column define
print(f'{"="*20} start column define {"="*20}')
feature_col = ['PART', 'CUREQP', 'PRE1EQP', 'PRE2EQP', 'RETICLE','PRERETICLE']
val_col = ['Tx_Rn', 'Ty_Rn']
X = train_dataset[feature_col]
y = train_dataset[val_col]

#preprocessing
#label encode & one hot encode
print(f'{"="*20} start label encode & one hot encode {"="*20}')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

le_list = []
for col in feature_col:
  le = LabelEncoder()
  X[col] = le.fit_transform(X[col])
  print(f'label encodinf done, {le.classes_}')
  le_list.append(le)
print(f'after label encoding, X is:\n{X.head()}')

ohe = OneHotEncoder()
X = ohe.fit_transform(X).toarray()
print(f'one hot encoder: {ohe.categories_}')
print(f'after one hot encoding, X is:\n{X}')

#get all permutations
print(f'{"="*20} start get all permutations {"="*20}')
index = list(range(len(X)))
permut = itertools.permutations(index,r=2)

#make y to numpy array
y = y.values

X_bias = np.empty((0,X.shape[1]))
y_bias = np.empty((0,y.shape[1]))
print(f'X:{type(X)},y:{type(y)}')
for pair in permut:
  temp_X = X[pair[0]] - X[pair[1]]
  temp_y = y[pair[0]] - y[pair[1]]
  X_bias = np.append(X_bias,temp_X)
  y_bias = np.append(y_bias,temp_y)

X_bias = X_bias.reshape(-1,X.shape[1])
y_bias = y_bias.reshape(-1,y.shape[1])
print(f'資料膨脹完畢')

#train / test split (注意這邊是把X_bias,y_bias拿去拆train test)
print(f'{"="*20} start train / test split {"="*20}')
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X_bias,y_bias , test_size=0.3 , random_state=40)
print(f'train/test split done, X_train , X_test , y_train , y_test size:\n{X_train.shape , X_test.shape , y_train.shape , y_test.shape}')
print(y_train[0:5])

#save to npz
np.savez('data' + '/OVL_tr.npz', features=X_train, labels=y_train)
np.savez('data' + '/OVL_te.npz', features=X_test, labels=y_test)



# pytorch會放在dataset裡面做
#Standardization (y train 做完scaler之後套給 y test)
# print(f'{"="*20} start Standardization {"="*20}')
# scaler = StandardScaler()
# y_train = scaler.fit_transform(y_train)
# print(f'after standardization, y_train is:\n{y_train[15:25]}')
# y_test = scaler.transform(y_test)