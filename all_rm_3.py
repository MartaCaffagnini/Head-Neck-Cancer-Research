import utility_nn as utl
import utility as utl_rf

import pandas as pd
import numpy as np
import sys
import os
from keras.utils.np_utils import to_categorical


# dataset directory
dataset_dir = 'datasets/features/'

# load patients with MRI images available
df_feat_mri = pd.read_csv(dataset_dir+'all_mri_radiomics_feat_AOP.csv')

# load patients with CT images available
df_feat_ct = pd.read_csv(dataset_dir+'all_ct_radiomics_feat_AOP.csv')

# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'openclinica_rs_ps_AOP.csv')

#elimino da dataframe i pazienti morti per altre cause
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_PR133'].index, inplace=True)
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_PR145'].index, inplace=True)
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_AOP_PR20_597'].index, inplace=True)



#Elimino i valori NaN da mri

#elimino le righe tutte vuote
df_feat_mri=df_feat_mri.drop('ROIType', 1) #elimino questa colonna che ha stringhe come valori
df_feat_mri=df_feat_mri.dropna(thresh=50) #elimino righe tutte vuote (righe con almeno 50 elementi non NaN


#Elimino le colonne di soli NaN da ct
df_feat_ct=df_feat_ct.dropna(axis=1, how='all')

#Unisco i due dataframe in uno unico
df_feat=pd.merge(left=df_feat_mri, right=df_feat_ct,how='outer', left_on="Patient_ID", right_on='BD2_ID')

#Estraggo dal dataframe dei dati clinici e pazienti del dataframe creato
df_openC = df_openC[ df_openC['Patient_ID'].isin(df_feat['Patient_ID']) ]

l_openC = list(df_openC.Patient_ID)

df_feat=df_feat.set_index('Patient_ID')
df_feat=df_feat.loc[~df_feat.index.duplicated(), :]
df_feat = df_feat.reindex(l_openC)
df_feat=df_feat.reset_index()

df_feat=df_feat.drop(df_feat.columns[1024],axis=1)

df_feat=df_feat.drop(columns=["Patient_ID"])
df_feat=df_feat.drop(df_feat.columns[0], axis=1)

df_feat=df_feat.fillna(0)


# create the training input tensor for mri images patients
# skip the gene labels in the first row
X = df_feat.to_numpy()
Y = utl.compute_follow_up(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

Y = to_categorical(Y)


# get the list of genes from oral cavity dataset
features = list(df_feat.columns)


k=6
num_val_samples = len(X) // k
all_scores = []
for i in range(k):
 print('processing fold #', i)
 val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
 val_targets = Y[i * num_val_samples: (i + 1) * num_val_samples]
 partial_train_data = np.concatenate( [X[:i * num_val_samples], X[(i + 1) * num_val_samples:]], axis=0)
 partial_train_targets = np.concatenate( [Y[:i * num_val_samples], Y[(i + 1) * num_val_samples:]], axis=0)
 model = utl.build_model(partial_train_data, 3)
 history=model.fit(partial_train_data, partial_train_targets, epochs=10, batch_size=2, verbose=0)
 _, acc = model.evaluate(val_data, val_targets, verbose=0)
 print(acc)
 all_scores.append(acc)
 
'''
for i in range(1,49):
 
 k=6
 num_val_samples = len(X) // k
 all_scores = []
 for i in range(k):
  print('processing fold #', i)
  val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
  val_targets = Y[i * num_val_samples: (i + 1) * num_val_samples]
  partial_train_data = np.concatenate( [X[:i * num_val_samples], X[(i + 1) * num_val_samples:]], axis=0)
  partial_train_targets = np.concatenate( [Y[:i * num_val_samples], Y[(i + 1) * num_val_samples:]], axis=0)
  model = utl.build_model(partial_train_data, 3)
  history=model.fit(partial_train_data, partial_train_targets, epochs=10, batch_size=2, verbose=0)
  _, acc = model.evaluate(val_data, val_targets, verbose=0)
  print(acc)
  all_scores.append(acc)
'''
model_rf, weight, acc_result= utl.rf_approach_3_stati(X,Y,features)
acc_rf=np.array(acc_result)

for i in range(1,99):
 classifier, weight, acc_result = utl.rf_approach_3_stati(X,Y,features)
 acc_rf =np.append( acc_rf ,acc_result)
 
av_acc_rf = np.average(acc_rf)
 
print(f"NN Average Accuracy Radiomic Data: {np.mean(all_scores)}")
print(f"RF Average Accuracy Radiomic Data: {av_acc_rf}")

