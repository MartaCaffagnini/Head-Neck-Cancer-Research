#MODELLI DI PREVISIONE NN E RF CON TUTTI I DATI GENOMICI. PREVISIONE A 2 STATI

import utility_nn as utl
import utility as utl_rf

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

df_feat=pd.merge(left=df_feat_oc, right=df_feat_op,how='outer', left_on="Unnamed: 0.1", right_on='Unnamed: 0.1')

df_feat=df_feat.set_index('Unnamed: 0.1')
df_feat = df_feat.transpose()
df_feat=df_feat.drop('Unnamed: 0_x', 0)
df_feat=df_feat.reset_index()
df_feat=df_feat.rename(columns={'index': 'Patient_ID'})
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
#df_feat=df_feat.drop(df_feat.columns[27971], axis=1)


#df_feat=df_feat.drop(df_feat.columns[0], axis=1)

df_feat=df_feat.fillna(0)


# create the training input tensor for mri images patients
# skip the gene labels in the first row
X = df_feat.to_numpy()
Y = utl.compute_alive_after(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

      

# get the list of genes from oral cavity dataset
features = list(df_feat.columns)

model, results, history= utl.nn_gm_approach(X,Y,features)
model_rf, weight, acc_result= utl.rf_approach(X,Y,features)

'''
#calcolo accuratezza media
acc_gm = np.array(results[1])
acc_rf=np.array(acc_result)

for i in range(1,99):
 model, results, history= utl.nn_gm_approach(X,Y,features)
 acc_gm =np.append( acc_gm ,results[1])
 classifier, weight, acc_result = utl.rf_approach(X,Y,features)
 acc_rf =np.append( acc_rf ,acc_result)
 

av_gm = np.average(acc_gm)
av_acc_rf = np.average(acc_rf)

print(f"NN Average Accuracy Genomic Data: {av_gm}")
print(f"RF Average Accuracy Genomic Data: {av_acc_rf}")
'''

#Calcolo geni piu importanti
def gene_weight(X,Y,features):
    print("Iteration 0")
    _, pesi,_ = utl.rf_approach_NO(X,Y,features)
    for i in range(99): #50
        print(f"Iteration {i+1}")
        _, weight,_ = utl.rf_approach_NO(X,Y,features)
        pesi = pd.concat([pesi, weight], axis=1)
    return pesi
   

pesi= gene_weight(X,Y,features)
media_pesi=pesi.mean(axis=1) #calcolo media valori per riga
media_pesi=media_pesi.sort_values( ascending=False) #metto valori in ordine decrescente
print("I 50 geni piu importanti sono: ")
print(media_pesi.iloc[:50])

