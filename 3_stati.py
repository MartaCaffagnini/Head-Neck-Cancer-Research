
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

# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'openclinica_rs_ps_AOP.csv')

#elimino da dataframe i pazienti morti per altre cause
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_PR133'].index, inplace=True)
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_PR145'].index, inplace=True)
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_AOP_PR20_597'].index, inplace=True)




#Elimino le righe tutte NaN da mri
df_feat_mri=df_feat_mri.drop('ROIType', 1) #elimino questa colonna che ha stringhe come valori
df_feat_mri=df_feat_mri.dropna(thresh=50) #elimino righe tutte vuote (righe con almeno 50 elementi non NaN

#Elimino le colonne di soli NaN da ct
df_feat_ct=df_feat_ct.dropna(axis=1, how='all')

# select the rows from openC relative to patients with MsRI patients
df_openC_mri = df_openC[ df_openC['Patient_ID'].isin(df_feat_mri['Patient_ID']) ]

# select the rows from openC relative to patients with CT images 
df_openC_ct = df_openC[ df_openC['Patient_ID'].isin(df_feat_ct['BD2_ID']) ]


Y_mri = utl.compute_follow_up(df_openC_mri,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

Y_mri = to_categorical(Y_mri)

# compute the 'alive after array' Y for CT patients
Y_ct = utl.compute_follow_up(df_openC_ct,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

Y_ct = to_categorical(Y_ct)

# get the list of MRI patients from openclinica records
l_mri_openC = list(df_openC_mri.Patient_ID)

# get the list of CT patients from openclinica records
l_ct_openC = list(df_openC_ct.Patient_ID)

# rearrange columns from MRI images dataset with the order given by l_mri_openC.
# This esures the same ordering of patients between openclinica mri dataset and
# mri features dataset.
# The index column (i.e., df_feat_mri.columns[0] will be dropped)
df_feat_mri=df_feat_mri.set_index('Patient_ID')
df_feat_mri=df_feat_mri.loc[~df_feat_mri.index.duplicated(), :]
df_feat_mri = df_feat_mri.reindex(l_mri_openC)
df_feat_mri=df_feat_mri.fillna(0)

# rearrange columns from CT dataset with the order given by l_ct_openC.
# This ensures the same ordering of patients between openclinica ct dataset and
# ct features dataset.
# The index column (i.e., df_feat_op.columns[0] will be dropped)
df_feat_ct=df_feat_ct.set_index('BD2_ID')

df_feat_ct = df_feat_ct.reindex(l_ct_openC)

df_feat_ct=df_feat_ct.fillna(0)


# create the training input tensor for mri images patients
# skip the gene labels in the first row
X_mri = df_feat_mri.to_numpy()

# create the training input tensor for ct patients
# skip the gene labels in the first row
X_ct = df_feat_ct.to_numpy()

# get the list of genes from oral cavity dataset
mri_features = list(df_feat_mri.columns)

# get the list of genes from oropharynx dataset
ct_features = list(df_feat_ct.columns)

# train the neural network model on mri images patients
model_mri, results_mri, history_mri= utl.tre_stati_rm_mri_approach(X_mri,Y_mri,mri_features)

# train the neural network model on ct images patients
model_ct, results_ct, history_ct = utl.tre_stati_rm_ct_approach(X_ct,Y_ct,ct_features)
'''
classifier_mri, weight_mri, result_rf_mri = utl.rf_rm_approach(X_mri,Y_mri,mri_features)
classifier_ct, weight_ct, result_rf_ct = utl.rf_approach(X_ct,Y_ct,ct_features)
'''
#Calcolo accuratezza media
acc_mri = np.array(results_mri[1])
acc_ct = np.array(results_ct[1])
'''
acc_rf_mri=np.array(result_rf_mri)
acc_rf_ct=np.array(result_rf_ct)
'''
for i in range(1,99):
 model_mri, results_mri, history_mri= utl.tre_stati_rm_mri_approach(X_mri,Y_mri,mri_features)
 acc_mri =np.append( acc_mri ,results_mri[1])
 model_ct, results_ct, history_ct = utl.tre_stati_rm_ct_approach(X_ct,Y_ct,ct_features)
 acc_ct =np.append( acc_ct ,results_ct[1])
 '''
 classifier_mri, weight_mri, result_rf_mri = utl.rf_rm_approach(X_mri,Y_mri,mri_features)
 acc_rf_mri =np.append( acc_rf_mri ,result_rf_mri)
 classifier_ct, weight_ct, result_rf_ct = utl.rf_approach(X_ct,Y_ct,ct_features)
 acc_rf_ct =np.append( acc_rf_ct ,result_rf_ct)
 '''
 
av_mri = np.average(acc_mri)
av_ct = np.average(acc_ct)
'''
av_mri_rf = np.average(acc_rf_mri)
av_ct_rf = np.average(acc_rf_ct)
'''
print(f"NN Average Accuracy MRI: {av_mri}")
print(f"NN Average Accuracy CT: {av_ct}")
'''
print(f"RF Average Accuracy MRI: {av_mri_rf}")
print(f"RF Average Accuracy CT: {av_ct_rf}")
'''
'''
#Calcolare feature piu importanti
def gene_weight1(X,Y,features):
    print("Iteration 0")
    _, pesi,_ = utl.rf_approach_NO(X,Y,features)
    for i in range(99): 
        print(f"Iteration {i+1}")
        _, weight,_ = utl.rf_approach_NO(X,Y,features)
        pesi = pd.concat([pesi, weight], axis=1) #aggiungo a pesi la colonna con i nuovi pesi calcolati
    return pesi
    
def gene_weight2(X,Y,features):
    print("Iteration 0")
    _, pesi,_ = utl.rf_rm_approach_NO(X,Y,features)
    for i in range(99):
        print(f"Iteration {i+1}")
        _, weight,_ = utl.rf_rm_approach_NO(X,Y,features)
        pesi = pd.concat([pesi, weight], axis=1)
        
    return pesi

pesi_mri = gene_weight1(X_mri,Y_mri,mri_features)
pesi_ct = gene_weight2(X_ct,Y_ct,ct_features) 

#calcolo media valori per riga
media_pesi_mri=pesi_mri.mean(axis=1) 
#metto valori in ordine decrescente
media_pesi_mri=media_pesi_mri.sort_values( ascending=False) 
print("Le 50 feature mri piu importanti sono: ")
print(media_pesi_mri.iloc[:50])
print("Elenco feature mri: ")
print(media_pesi_mri.iloc[:50].index)


media_pesi_ct=pesi_ct.mean(axis=1)
media_pesi_ct=media_pesi_ct.sort_values( ascending=False)
print("Le 50 feature ct piu importanti sono: ")
print(media_pesi_ct.iloc[:50])
print("Elenco feature ct: ")
print(media_pesi_ct.iloc[:50].index)
'''