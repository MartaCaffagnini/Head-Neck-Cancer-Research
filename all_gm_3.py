import utility_nn as utl
import utility as utl_rf

import pandas as pd
import numpy as np
import sys
import os
from keras.utils.np_utils import to_categorical



# dataset directory
dataset_dir = 'datasets/features/'

# load the oral cavity patients
df_feat_oc = pd.read_csv(dataset_dir+'oc_genomics_feat_AOP.csv')

# load the oropharynx patients
df_feat_op = pd.read_csv(dataset_dir+'op_genomics_feat_AOP.csv')

# load the openclinica medical records
df_openC = pd.read_csv(dataset_dir+'openclinica_rs_ps_AOP.csv')

df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_PR133'].index, inplace=True)
df_openC.drop(df_openC[df_openC['Patient_ID'] =='SS_AOP_PR20_597'].index, inplace=True)

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
Y = utl.compute_follow_up(df_openC,start='clinical_Date_of_first_Diagnosis',
end='follow_Date_of_Death',alive_after=5*365)

Y = to_categorical(Y)

# get the list of genes from oral cavity dataset
features = list(df_feat.columns)

k=6
num_val_samples = len(X) // k
all_scores = []
for i in range(k):
 print('processing fold #', i)
 val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
 val_targets = Y[i * num_val_samples: (i + 1) * num_val_samples]
 partial_train_data = np.concatenate( [X[:i * num_val_samples], X[(i + 1) * num_val_samples:]], axis=0)
 partial_train_targets = np.concatenate( [Y[:i * num_val_samples], Y[(i + 1) * num_val_samples:]], axis=0)
 model = utl.build_model(partial_train_data, 16)
 history=model.fit(partial_train_data, partial_train_targets, epochs=10, batch_size=2, verbose=0)
 _, acc = model.evaluate(val_data, val_targets, verbose=0)
 print(acc)
 all_scores.append(acc)
 
model_rf, weight, acc_result= utl.rf_approach_3_stati(X,Y,features)



for i in range(1,49):
  
 for i in range(k):
  print('processing fold #', i)
  val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
  val_targets = Y[i * num_val_samples: (i + 1) * num_val_samples]
  partial_train_data = np.concatenate( [X[:i * num_val_samples], X[(i + 1) * num_val_samples:]], axis=0)
  partial_train_targets = np.concatenate( [Y[:i * num_val_samples], Y[(i + 1) * num_val_samples:]], axis=0)
  model = utl.build_model(partial_train_data,3)
  history=model.fit(partial_train_data, partial_train_targets, epochs=12, batch_size=2, verbose=0)
  _, acc = model.evaluate(val_data, val_targets, verbose=0)
  all_scores.append(acc) 

model_rf, weight, acc_result= utl.rf_approach_3_stati(X,Y,features)
acc_rf=np.array(acc_result)

for i in range(1,99):
 classifier, weight, acc_result = utl.rf_approach_3_stati(X,Y,features)
 acc_rf =np.append( acc_rf ,acc_result)
 
av_acc_rf = np.average(acc_rf)
 
print(f"NN Average Accuracy Radiomic Data: {np.mean(all_scores)}")
print(f"RF Average Accuracy Radiomic Data: {av_acc_rf}")


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

'''
