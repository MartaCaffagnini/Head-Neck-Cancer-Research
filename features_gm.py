#MODELLI DI PREVISIONE NN E RF CON OC E OP SEPARATI, PREVISIONE A 2 STATI

import utility_nn as utl
import pandas as pd
import numpy as np
import sys
import os

# dataset directory
dataset_dir = 'datasets/features/'

# load the oral cavity patients
df_feat_oc = pd.read_csv(dataset_dir+'oc_genomics_feat_AOP.csv')

# load the oropharynx patients
df_feat_op = pd.read_csv(dataset_dir+'op_genomics_feat_AOP.csv')

# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'openclinica_rs_ps_AOP.csv')

# select the rows from openC relative to oral cavity patients
df_openC_oc = df_openC[ df_openC['Patient_ID'].isin(df_feat_oc.columns[2:]) ]

# select the rows from openC relative to oropharynx cases
df_openC_op = df_openC[ df_openC['Patient_ID'].isin(df_feat_op.columns[2:]) ]

# compute the 'alive after array' Y for oral cavity patients
Y_oc = utl.compute_alive_after(df_openC_oc,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

# compute the 'alive after array' Y for oropharynx patients
Y_op = utl.compute_alive_after(df_openC_op,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

# get the list of 'oral cavity' patients from openclinica records
l_oc_openC = list(df_openC_oc.Patient_ID)

# update the list with the gene column key from oral cavity dataset
l_oc_openC.insert(0,'Unnamed: 0.1')

# get the list of 'oropharynx' patients from openclinica records
l_op_openC = list(df_openC_op.Patient_ID)

# update the list with the gene column key from oropharynx dataset
l_op_openC.insert(0,'Unnamed: 0.1')

# rearrange columns from oral cavity dataset with the order given by l_oc_openC.
# This esures the same ordering of patients between openclinica oc dataset and
# oral cavity features dataset.
# The index column (i.e., df_feat_oc.columns[0] will be dropped)
df_feat_oc = df_feat_oc.reindex(columns=l_oc_openC)

# rearrange columns from oropharynx dataset with the order given by l_op_openC.
# This ensures the same ordering of patients between openclinica op dataset and
# oropharynx features dataset.
# The index column (i.e., df_feat_op.columns[0] will be dropped)
df_feat_op = df_feat_op.reindex(columns=l_op_openC)

# transpose the rearranged oral cavity dataset
df_feat_oc = df_feat_oc.transpose()

# transpose the rearranged oropharynx dataset
df_feat_op = df_feat_op.transpose()

# create the training input tensor for oral cavity patients
# skip the gene labels in the first row
X_oc = df_feat_oc[1:].to_numpy()

# create the training input tensor for oropharynx patients
# skip the gene labels in the first row
X_op = df_feat_op[1:].to_numpy()

# get the list of genes from oral cavity dataset
oc_features = list(df_feat_oc.iloc[0])

# get the list of genes from oropharynx dataset
op_features = list(df_feat_op.iloc[0])

# train the random forest model on oral cavity patients
model_oc, results_oc, history_oc = utl.nn_oc_approach(X_oc,Y_oc,oc_features)

# train the random forest model on oropharynx patients
model_op, results_op, history_op = utl.nn_op_approach(X_op,Y_op,op_features)

classifier_oc, weight_oc, result_rf_oc = utl.rf_approach(X_oc,Y_oc,oc_features)

classifier_op, weight_op, result_rf_op = utl.rf_approach(X_op,Y_op,op_features)

#Calcolare accuratezza media
'''
acc_oc = np.array(results_oc[1])
acc_op = np.array(results_op[1])
acc_rf_oc=np.array(result_rf_oc)
acc_rf_op=np.array(result_rf_op)

for i in range(1,99):
 model_oc, results_oc, history_oc = utl.nn_oc_approach(X_oc,Y_oc,oc_features)
 acc_oc =np.append( acc_oc ,results_oc[1])
 model_op, results_op, history_op = utl.nn_op_approach(X_op,Y_op,op_features)
 acc_op =np.append( acc_op ,results_op[1])

 classifier_oc, weight_oc, result_rf_oc = utl.rf_approach(X_oc,Y_oc,oc_features)
 acc_rf_oc =np.append( acc_rf_oc ,result_rf_oc)
 
 classifier_op, weight_op, result_rf_op = utl.rf_approach(X_op,Y_op,op_features)
 acc_rf_op =np.append( acc_rf_op ,result_rf_op)
 
 
av_oc = np.average(acc_oc)
av_op = np.average(acc_op)
av_oc_rf = np.average(acc_rf_oc)
av_op_rf = np.average(acc_rf_op)

print(f"NN Average Accuracy Oral Cavity: {av_oc}")
print(f"NN Average Accuracy Oropharynx: {av_op}")
print(f"RF Average Accuracy Oral Cavity: {av_oc_rf}")
print(f"RF Average Accuracy Oropharynx: {av_op_rf}")
'''
#Calcolare geni piu importanti

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

pesi_oc = gene_weight1(X_oc,Y_oc,oc_features)
pesi_op = gene_weight2(X_op,Y_op,op_features) 

media_pesi_oc=pesi_oc.mean(axis=1) #calcolo media valori per riga
media_pesi_oc=media_pesi_oc.sort_values( ascending=False) #metto valori in ordine decrescente
print("I 50 geni oc piu importanti sono: ")
print(media_pesi_oc.iloc[:50])

media_pesi_op=pesi_op.mean(axis=1)
media_pesi_op=media_pesi_op.sort_values( ascending=False)
print("I 50 geni op piu importanti sono: ")
print(media_pesi_op.iloc[:50])
print("Elenco geni op: ")
print(media_pesi_op.iloc[:50].index)


