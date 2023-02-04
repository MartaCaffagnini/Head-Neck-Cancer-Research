# MODELLO CON TUTTI I DATI RADIOMICI, GENOMICI. PREVISIONE A DUE STATI
import utility_nn as utl

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

# load the oral cavity patients
df_feat_oc = pd.read_csv(dataset_dir+'oc_genomics_feat_AOP.csv')

# load the oropharynx patients
df_feat_op = pd.read_csv(dataset_dir+'op_genomics_feat_AOP.csv')

# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'openclinica_rs_ps_AOP.csv')

#tolgo NaN
df_feat_mri=df_feat_mri.drop('ROIType', 1) #elimino questa colonna che ha stringhe come valori
df_feat_mri=df_feat_mri.dropna(thresh=50) #elimino righe tutte vuote (righe con almeno 50 elementi non NaN
df_feat_ct=df_feat_ct.dropna(axis=1, how='all')

df_feat=pd.merge(left=df_feat_oc, right=df_feat_op,how='outer', left_on="Unnamed: 0.1", right_on='Unnamed: 0.1')

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
Y = utl.compute_alive_after(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

      

# get the list of genes from oral cavity dataset
features = list(df_feat.columns)

model, results, history= utl.nn_feat_approach(X,Y,features)
classifier, weight, acc_result = utl.rf_approach(X,Y,features)

acc = np.array(results[1])
acc_rf=np.array(acc_result)
for i in range(1,99):
 model, results, history= utl.nn_feat_approach(X,Y,features)
 acc =np.append( acc ,results[1])
 classifier, weight, acc_result = utl.rf_approach(X,Y,features)
 acc_rf =np.append( acc_rf ,acc_result)
 

av_acc = np.average(acc)
av_acc_rf = np.average(acc_rf)

print(f"NN Average Accuracy {av_acc}")
print(f"RF Average Accuracy {av_acc_rf}")

'''
def gene_intersections(X,Y,features,inters_length):

    iters = []
    print("Computing intersection sets ...")

    # prepare iterate the training a number of times and save results
    for i in range(100):
        print(f"Iteration {i}")
        _, weight,_ = utl.rf_approach(X,Y,features)
        iters.append(set(weight.index[0:inters_length]))


    # build the intersections to check which genes appare in multiple
    # runs. The goal is to look for genes having a certain importance
    # (weight) in the models produces
    # intersections can also be performed in the loop above directly..
    # for more performance

    intersection_set = iters[0]
    for it in iters[1:]:
        intersection_set = intersection_set.intersection(it)

    return intersection_set
    

intersection_set = gene_intersections(X,Y,features,12000)


print("Intersection set (gm) on first 12000 genes, 100 iterations:",intersection_set) #prima 12000
print("size:",len(intersection_set))
'''




#PLOT 
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.clf() # clear figure

#prendo i valori dal dizionario
history_dict = history.history

acc = history_dict['accuracy']

#val_acc = history_dict['val_accuracy']
loss = history_dict['loss']

#val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label = 'Training Loss Gemomics and Radiomics Features') 

plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('feat_loss.png')

plt.clf() # clear figure

plt.plot(epochs, acc, 'r', label='Training Accuracy Gemomics and Radiomics Features')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('feat_accuracy.png')
'''
