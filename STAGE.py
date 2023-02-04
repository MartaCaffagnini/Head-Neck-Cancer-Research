#CALCOLO ACCURATEZZA DEGLI STADI USANDO CONDIZIONE PAZIENTE A 5 ANNI
import utility as utl
import pandas as pd
import numpy as np
import sys
import os

# dataset directory
dataset_dir = 'datasets/'

# load patients with TNM
df = pd.read_csv(dataset_dir+'openclinica_rs_ALL_extended.csv',sep=';')
# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'features/openclinica_rs_ps_AOP.csv')

df=df[['Patient_ID','ctn_Stage_at_Diagnosis_7Edition']]

'''
#TNM ottava edizione
df=df[['Patient_ID','ctn_TNM_cT_8Edition_OralCavity_Oropharynx_p16Negative_Hypopharynx_Larynx']]
#Drop the rows where at least one element is missing
df=df.dropna()
'''

df_openC = df_openC[ df_openC['Patient_ID'].isin(df['Patient_ID']) ]

# compute the 'alive after array' Y 
Y = utl.compute_alive_after(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

# get the list of MRI patients from openclinica records
l_openC = list(df_openC.Patient_ID)

# rearrange columns from MRI images dataset with the order given by l_mri_openC.
# This esures the same ordering of patients between openclinica mri dataset and
# mri features dataset.
# The index column (i.e., df_feat_mri.columns[0] will be dropped)
df=df.set_index('Patient_ID')
df = df.reindex(l_openC)

X = df.to_numpy()

#rischio basso 1 (vivo), elevato 0 (come morto)
result=X
for index,i in enumerate(X):
 if ((i=='Stage III') or (i== 'Stage I')):
    result[index]=1
 elif ((i=='Stage IVA') or (i=='Stage IVB')):
    result[index]=0

result=np.transpose(result)
#pazienti totali
totali=result.size
#numero output errati
errori=np.sum(np.abs(result -Y)) 

#output corretti
corretti=totali-errori

#accuratezza
acc=corretti/totali
print(f'Accuratezza: {acc}')
#Accuratezza: 0.63


