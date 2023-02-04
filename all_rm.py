#MODELLO NN E RF CON TUTTI I DATI RADIOMICI. PREVISIONE A 2 STATI
import utility_nn as utl
import utility as utl_rf

import pandas as pd
import numpy as np
import sys
import os

# dataset directory
dataset_dir = 'datasets/features/'

# load patients with MRI images available
df_feat_mri = pd.read_csv(dataset_dir+'all_mri_radiomics_feat_AOP.csv')

# load patients with CT images available
df_feat_ct = pd.read_csv(dataset_dir+'all_ct_radiomics_feat_AOP.csv')

# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'openclinica_rs_ps_AOP.csv')

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
Y = utl.compute_alive_after(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

      

# get the list of genes from oral cavity dataset
features = list(df_feat.columns)


# train the neural network model on mri images patients
model, results, history= utl.nn_rm_approach(X,Y,features)
model_rf, weight, acc_result= utl.rf_rm_approach(X,Y,features)


acc_rm = np.array(results[1])
acc_rf=np.array(acc_result)

for i in range(1,99):
 model, results, history= utl.nn_rm_approach(X,Y,features)
 acc_rm =np.append( acc_rm ,results[1])
 classifier, weight, acc_result = utl.rf_rm_approach(X,Y,features)
 acc_rf =np.append( acc_rf ,acc_result)
 

av_rm = np.average(acc_rm)
av_acc_rf = np.average(acc_rf)

print(f"NN Average Accuracy Radiomic Data: {av_rm}")
print(f"RF Average Accuracy Radiomic Data: {av_acc_rf}")
