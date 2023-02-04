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


#preparo dati clinici di AOP
colonna_ID=utl.get_rs_full_dataframe()
colonna_ID=colonna_ID[0:92]
colonna_ID=colonna_ID[['Patient_ID']]

df_AOP=utl.get_rs_dataframe()
df_AOP=df_AOP[0:92]
df_AOP.insert(0,"Patient_ID",colonna_ID) #aggiungo la colonna con tutti gli ID

#preparo dati genomici oc
df_feat_oc=df_feat_oc.set_index('Unnamed: 0.1')
df_feat_oc = df_feat_oc.transpose()
df_feat_oc=df_feat_oc.reset_index()
df_feat_oc=df_feat_oc.rename(columns={'index': 'Patient_ID'})

#preparo dati genomici op
df_feat_op=df_feat_op.set_index('Unnamed: 0.1')
df_feat_op = df_feat_op.transpose()
df_feat_op=df_feat_op.reset_index()
df_feat_op=df_feat_op.rename(columns={'index': 'Patient_ID'})


df_feat_oc=pd.merge(left=df_feat_oc, right=df_AOP,how='outer', left_on="Patient_ID", right_on='Patient_ID')

df_feat_op=pd.merge(left=df_feat_op, right=df_AOP,how='outer', left_on="Patient_ID", right_on='Patient_ID')

#ho tutti i pazienti, ora li metto in ordine
df_openC = df_openC[ df_openC['Patient_ID'].isin(df_feat_oc['Patient_ID']) ]
df_openC = df_openC[ df_openC['Patient_ID'].isin(df_feat_op['Patient_ID']) ]

# get the list of 'oral cavity' patients from openclinica records
l_openC = list(df_openC.Patient_ID)


df_feat_oc=df_feat_oc.set_index('Patient_ID')
df_feat_oc=df_feat_oc.loc[~df_feat_oc.index.duplicated(), :]
df_feat_oc = df_feat_oc.reindex(l_openC)
df_feat_oc=df_feat_oc.reset_index()

df_feat_op=df_feat_op.set_index('Patient_ID')
df_feat_op=df_feat_op.loc[~df_feat_op.index.duplicated(), :]
df_feat_op = df_feat_op.reindex(l_openC)
df_feat_op=df_feat_op.reset_index()

df_feat_oc=df_feat_oc.drop(columns=["Patient_ID"])
df_feat_oc=df_feat_oc.fillna(0)

df_feat_op=df_feat_op.drop(columns=["Patient_ID"])
df_feat_op=df_feat_op.fillna(0)


# create the training input tensor for mri images patients
# skip the gene labels in the first row
X_oc = df_feat_oc.to_numpy()
Y_oc = utl.compute_alive_after(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

X_op = df_feat_op.to_numpy()
Y_op = utl.compute_alive_after(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

# get the list of genes from oral cavity dataset
oc_features = list(df_feat_oc.columns)
op_features= list(df_feat_op.columns)

model_oc, results_oc, history_oc = utl.nn_cl_gm_oc_approach(X_oc,Y_oc,oc_features)

# train the random forest model on oropharynx patients
model_op, results_op, history_op = utl.nn_cl_gm_op_approach(X_op,Y_op,op_features)

classifier_oc, weight_oc, result_rf_oc = utl.rf_approach(X_oc,Y_oc,oc_features)

classifier_op, weight_op, result_rf_op = utl.rf_approach(X_op,Y_op,op_features)

#Calcolare accuratezza media

acc_oc = np.array(results_oc[1])
acc_op = np.array(results_op[1])
acc_rf_oc=np.array(result_rf_oc)
acc_rf_op=np.array(result_rf_op)

for i in range(1,99):
 
 model_oc, results_oc, history_oc = utl.nn_cl_gm_oc_approach(X_oc,Y_oc,oc_features)
 acc_oc =np.append( acc_oc ,results_oc[1])
 model_op, results_op, history_op = utl.nn_cl_gm_op_approach(X_op,Y_op,op_features)
 acc_op =np.append( acc_op ,results_op[1])
 '''
 classifier_oc, weight_oc, result_rf_oc = utl.rf_approach(X_oc,Y_oc,oc_features)
 acc_rf_oc =np.append( acc_rf_oc ,result_rf_oc)
 
 classifier_op, weight_op, result_rf_op = utl.rf_approach(X_op,Y_op,op_features)
 acc_rf_op =np.append( acc_rf_op ,result_rf_op)
 '''

av_oc = np.average(acc_oc)
av_op = np.average(acc_op)
'''
av_oc_rf = np.average(acc_rf_oc)
av_op_rf = np.average(acc_rf_op)
'''
print(f"NN Average Accuracy Oral Cavity: {av_oc}")
print(f"NN Average Accuracy Oropharynx: {av_op}")
'''
print(f"RF Average Accuracy Oral Cavity: {av_oc_rf}")
print(f"RF Average Accuracy Oropharynx: {av_op_rf}")
'''



    
