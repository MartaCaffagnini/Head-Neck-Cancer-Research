import utility_nn as utl
import new_utl as new
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

acc_oc = np.array(results_oc[1])
acc_op = np.array(results_op[1])
acc_rf_oc=np.array(result_rf_oc)
acc_rf_op=np.array(result_rf_op)

for i in range(1,99):
 #model_oc, results_oc, history_oc = utl.nn_oc_approach(X_oc,Y_oc,oc_features)
 #acc_oc =np.append( acc_oc ,results_oc[1])
 #model_op, results_op, history_op = utl.nn_op_approach(X_op,Y_op,op_features)
 #acc_op =np.append( acc_op ,results_op[1])

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

#Calcolare geni piu importanti
'''
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
'''

#PLOT
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.clf() # clear figure

#prendo i valori dal dizionario
history_dict_oc = history_oc.history
history_dict_op = history_op.history

acc_oc = history_dict_oc['accuracy']
acc_op = history_dict_op['accuracy']

#val_acc = history_dict['val_accuracy']
loss_oc = history_dict_oc['loss']
loss_op = history_dict_op['loss']

#val_loss = history_dict['val_loss']

epochs_oc = range(1, len(acc_oc) + 1)
epochs_op = range(1, len(acc_op) + 1)

plt.plot(epochs_oc, loss_oc, 'r', label = 'Training Loss Oral Cavity') 
plt.plot(epochs_op, loss_op, 'b', label = 'Training Loss Oropharynx') 
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('feat_gm_loss.png')

plt.clf() # clear figure

plt.plot(epochs_oc, acc_oc, 'r', label='Training Accuracy Oral Cavity')
plt.plot(epochs_op, acc_op, 'b', label='Training Accuracy Oropharynx')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('feat_gm_accuracy.png')
'''
#####################################
'''
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%Y%m%d%H%M%S")
txtfile="test_4_"+date_time+".txt"
pngfile="test_4_"+date_time+".png"

infile = open(txtfile,"w")
for i in range(1000):
   myrow = str(i+1) + " " +  weight_oc.index[i] + " " +  str(weight_oc.iloc[i,0]) + "\n"
   infile.write(myrow)
infile.close()


#####################################

import matplotlib
matplotlib.use('Agg')   # Backend per PNG
import matplotlib.pyplot as plt

df = pd.read_csv(txtfile, sep=' ', names=["IDX", "NAM", "WGT"])
df.index = df.index + 1

#print(df.head(5))

IDX=(df[df.columns[0]])
NAM=(df[df.columns[1]])
WGT=(df[df.columns[2]])

#print (str(NAM.iloc[:10]))

plt.figure(1) #
plt.grid()
plt.title('OC weight')
plt.xlabel('index')
plt.ylabel('weight')
plt.plot(IDX[:50],  WGT[:50],'-o', label='OC weight')
plt.text(30, 0.0006, str(NAM.iloc[:20]))

plt.savefig(pngfile)


def gene_intersections(X,Y,features,inters_length):

    iters = []
    print("Computing intersection sets ...")

    # prepare iterate the training a number of times and save results
    for i in range(10):
        print(f"Iteration {i}")
        _, weight = utl.rf_approach(X,Y,features)
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

intersection_set_oc = gene_intersections(X_oc,Y_oc,oc_features,1000)

print("Intersection set (oc) on first 1000 genes, 10 iterations:",intersection_set_oc)
print("size:",len(intersection_set_oc))

#
'''
