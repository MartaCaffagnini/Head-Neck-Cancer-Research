'''
prendere tutti i pazienti di features/openclinica_rs_ps_AOP.csv
suddividere i pazienti in base al follow up nelle 3 categorie, i morti per altre cause li eliminiamo
per ogni categoria vedere il tnm (chiedere se guardare solo il T), stadio
e fare le percentuali

'''
#CALCOLO ACCURATEZZA TNM USANDO CONDIZIONE PAZIENTE A 5 ANNI
import utility_nn as utl
import pandas as pd
import numpy as np
import sys
import os


# dataset directory
dataset_dir = 'datasets/'

df_openC = pd.read_csv(dataset_dir+'features/openclinica_rs_ps_AOP.csv')

df_openC = df_openC.drop(index=[70,76,104])

Y = utl.compute_follow_up(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

#df_openC=df_openC[['Patient_ID','ctn_TNM_cT_7Edition','ctn_Stage_at_Diagnosis_7Edition','chemo_Cancer_Therapy_Agent_Name','chemo_Chemotherapy_Treatment','surge_Type_of_Surgery','surge_Kind_of_Left_Neck_Dissection','surge_Kind_of_Right_Neck_Dissection', 'surge_Site_Neck_Dissection']]
#df_openC=df_openC[['ctn_TNM_cT_7Edition','ctn_Stage_at_Diagnosis_7Edition','chemo_Cancer_Therapy_Agent_Name','chemo_Chemotherapy_Treatment','surge_Type_of_Surgery','surge_Kind_of_Left_Neck_Dissection','surge_Kind_of_Right_Neck_Dissection', 'surge_Site_Neck_Dissection']]
df_openC=df_openC[['ctn_TNM_cT_7Edition','ctn_Stage_at_Diagnosis_7Edition','chemo_Chemotherapy_Treatment','surge_Type_of_Surgery','surge_Kind_of_Left_Neck_Dissection','surge_Kind_of_Right_Neck_Dissection','surge_Site_Neck_Dissection']]


def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = 100
    display.max_rows = 100
    display.max_colwidth = 199
    display.width = None

set_pandas_display_options()

'''
#TNM ottava edizione
df=df[['Patient_ID','ctn_TNM_cT_8Edition_OralCavity_Oropharynx_p16Negative_Hypopharynx_Larynx']]
#Drop the rows where at least one element is missing
df=df.dropna()
'''

df_openC['Follow_up'] = Y
print ("Suddivisione in base allo stadio di tutti i pazienti")
III_tot,IVA_tot,IVB_tot= utl.suddivisione_stadi(df_openC)


df=df_openC['Follow_up']==1
#df con tutti i pazienti vivi senza recidiva
df_AND = df_openC[df] 
print ("Suddivisione in base al TNM dei pazienti vivi senza recidive")
utl.suddivisione_TNM(df_AND)
print ("Suddivisione in base allo stadio dei pazienti vivi senza recidive")
III_AND,IVA_AND,IVB_AND=utl.suddivisione_stadi(df_AND)

df=df_openC['Follow_up']==0
#df con tutti i pazienti morti per malattia
df_DOD = df_openC[df] 
print ("Suddivisione in base al TNM dei pazienti morti per malattia")
utl.suddivisione_TNM(df_DOD)
print ("Suddivisione in base allo stadio dei pazienti morti per malattia")
III_DOD,IVA_DOD,IVB_DOD=utl.suddivisione_stadi(df_DOD)

df=df_openC['Follow_up']==2
#df con tutti i pazienti vivi con recidiva
df_AWD = df_openC[df] 
print ("Suddivisione in base al TNM dei pazienti vivi con recidiva")
utl.suddivisione_TNM(df_AWD)
print ("Suddivisione in base allo stadio dei vivi con recidiva")
III_AWD,IVA_AWD,IVB_AWD=utl.suddivisione_stadi(df_AWD)

print('Percentuali dei pazienti con stadio III')
print('AND: ', (III_AND*100)/III_tot)
print('DOD: ', (III_DOD*100)/III_tot)
print('AWD: ', (III_AWD*100)/III_tot)
print('Percentuali dei pazienti con stadio IVa')
print('AND: ', (IVA_AND*100)/IVA_tot)
print('DOD: ', (IVA_DOD*100)/IVA_tot)
print('AWD: ', (IVA_AWD*100)/IVA_tot)
print('Percentuali dei pazienti con stadio IVb')
print('AND: ', (IVB_AND*100)/IVB_tot)
print('DOD: ', (IVB_DOD*100)/IVB_tot)
print('AWD: ', (IVB_AWD*100)/IVB_tot)

'''
print ('TABELLA AND')
df_AND=df_AND.drop('Follow_up', 1) 
stampa=df_AND
stampa.columns.values[0] = "TNM" #settima edizione
stampa.columns.values[1] = "Stage"
stampa.columns.values[2] = "CT"
stampa.columns.values[3] = "surge_Type_of_Surgery"
stampa.columns.values[4] = "L_N_Diss"
stampa.columns.values[5] = "R_N_Diss"
stampa.columns.values[6] = "Site_Neck_Diss"
print (stampa.to_string(index=False))

print ('TABELLA DOD')
df_DOD=df_DOD.drop('Follow_up', 1) 
stampa=df_DOD
stampa.columns.values[0] = "TNM"
stampa.columns.values[1] = "Stage"
stampa.columns.values[2] = "Chemotherapy"
stampa.columns.values[3] = "surge_Type_of_Surgery"
stampa.columns.values[4] = "L_Neck_Diss"
stampa.columns.values[5] = "R_Neck_Diss"
stampa.columns.values[6] = "Site_Neck_Diss"
print (stampa.to_string(index=False))

print ('TABELLA AWD')
df_AWD=df_AWD.drop('Follow_up', 1) 
stampa=df_DOD
stampa.columns.values[0] = "TNM"
stampa.columns.values[1] = "Stage"
stampa.columns.values[2] = "Chemotherapy"
stampa.columns.values[3] = "surge_Type_of_Surgery"
stampa.columns.values[4] = "L_Neck_Diss"
stampa.columns.values[5] = "R_Neck_Diss"
stampa.columns.values[6] = "Site_Neck_Diss"
print (stampa.to_string(index=False))
'''


