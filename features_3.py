# MODELLO CON TUTTI I DATI RADIOMICI, GENOMICI
import utility_nn as utl

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

# load the oral cavity patients
df_feat_oc = pd.read_csv(dataset_dir+'oc_genomics_feat_AOP.csv')

# load the oropharynx patients
df_feat_op = pd.read_csv(dataset_dir+'op_genomics_feat_AOP.csv')

# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'openclinica_rs_ps_AOP.csv')

#elimino da dataframe i pazienti morti per altre cause
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_PR133'].index, inplace=True)
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_PR145'].index, inplace=True)
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_AOP_PR20_597'].index, inplace=True)

#tolgo NaN
df_feat_mri=df_feat_mri.drop('ROIType', 1) #elimino questa colonna che ha stringhe come valori
df_feat_mri=df_feat_mri.dropna(thresh=50) #elimino righe tutte vuote (righe con almeno 50 elementi non NaN
df_feat_ct=df_feat_ct.dropna(axis=1, how='all')

df_feat=pd.merge(left=df_feat_oc, right=df_feat_op,how='outer', left_on="Unnamed: 0.1", right_on='Unnamed: 0.1')

'''
# transpose the rearranged oral cavity dataset
df_feat_oc = df_feat_oc.transpose()

df_feat_oc=df_feat_oc.reset_index()
df_feat_oc.rename(columns={'index':'Patient_ID'}, inplace=True)

df_feat_op = df_feat_op.transpose()
df_feat_op=df_feat_op.reset_index()
df_feat_op.rename(columns={'index':'Patient_ID'}, inplace=True)
'''
#df_feat=pd.merge(left=df_feat_oc, right=df_feat_op,how='outer', left_on="Patient_ID", right_on='Patient_ID')


# compute the 'alive after array' Y for oral cavity patients
df_feat=df_feat.set_index('Unnamed: 0.1')
df_feat = df_feat.transpose()
df_feat=df_feat.drop('Unnamed: 0_x', 0)
df_feat=df_feat.reset_index()
df_feat=df_feat.rename(columns={'index': 'Patient_ID'})


#voglio tenere tutti i pazienti oc e op ed eventualmente aggiungere rmi e ct a quei pazienti
#df_feat_mri = df_feat_mri[ df_feat_mri['Patient_ID'].isin(df_feat['Patient_ID']) ]

#df_feat_ct = df_feat_ct[ df_feat_ct['BD2_ID'].isin(df_feat['Patient_ID']) ]


df_feat=pd.merge(left=df_feat, right=df_feat_mri,how='outer', left_on="Patient_ID", right_on='Patient_ID')
df_feat=pd.merge(left=df_feat, right=df_feat_ct,how='outer', left_on="Patient_ID", right_on='BD2_ID')

#ho tutti i pazienti, ora li metto in ordine
df_openC = df_openC[ df_openC['Patient_ID'].isin(df_feat['Patient_ID']) ]

# get the list of 'oral cavity' patients from openclinica records
l_openC = list(df_openC.Patient_ID)

# update the list with the gene column key from oral cavity dataset
#l_openC.insert(0,'Unnamed: 0.1')

df_feat=df_feat.set_index('Patient_ID')
df_feat=df_feat.loc[~df_feat.index.duplicated(), :]
df_feat = df_feat.reindex(l_openC)
df_feat=df_feat.reset_index()



#df_feat=df_feat.drop(df_feat.columns[0])
df_feat=df_feat.drop(columns=["Patient_ID"])
df_feat=df_feat.drop(df_feat.columns[27971], axis=1)


#df_feat=df_feat.drop(df_feat.columns[0], axis=1)

df_feat=df_feat.fillna(0)


# create the training input tensor for mri images patients
# skip the gene labels in the first row
X = df_feat.to_numpy()
Y = utl.compute_follow_up(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

Y = to_categorical(Y)

      

# get the list of genes from oral cavity dataset
features = list(df_feat.columns)

k=5
num_val_samples = len(X) // k
all_scores = []
for i in range(1,49):

 for i in range(k):
  print('processing fold #', i)
  val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
  val_targets = Y[i * num_val_samples: (i + 1) * num_val_samples]
  partial_train_data = np.concatenate( [X[:i * num_val_samples], X[(i + 1) * num_val_samples:]], axis=0)
  partial_train_targets = np.concatenate( [Y[:i * num_val_samples], Y[(i + 1) * num_val_samples:]], axis=0)
  model = utl.build_model(partial_train_data, 10)
  history=model.fit(partial_train_data, partial_train_targets, epochs=11, batch_size=2, verbose=0)
  _, acc = model.evaluate(val_data, val_targets, verbose=0)
  print(acc)
  all_scores.append(acc)

accuracy=np.mean(all_scores)

model_rf, weight, acc_result= utl.rf_approach_3_stati(X,Y,features)
acc_rf=np.array(acc_result)

for i in range(1,99):
 classifier, weight, acc_result = utl.rf_approach_3_stati(X,Y,features)
 acc_rf =np.append( acc_rf ,acc_result)
 
av_acc_rf = np.average(acc_rf)
 
print(f"NN Average Accuracy Radiomic Data: {np.mean(all_scores)}")
print(f"RF Average Accuracy Radiomic Data: {av_acc_rf}")
