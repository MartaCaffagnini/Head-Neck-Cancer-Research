# NN WITH CLINICAL DATA. PREVISIONE A 2 STATI
import utility_nn as utl 
import pandas as pd
import numpy as np 
from keras.utils.np_utils import to_categorical


# get training dataset
#X, Y, features = utl.get_rs_training_data_3_stati(alive_after=5*365)
X, Y, features = utl.get_rs_training_data_3_stati_AOP(alive_after=5*365)

# Train on Xs,Ys
Y = to_categorical(Y)

#k-fold validation
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
 
for i in range(1,99):
  
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
 

print(f"NN Average Accuracy Clinical Data: {np.mean(all_scores)}")
print(f"RF Average Accuracy Clinical Data: {av_acc_rf}")

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

pesi = gene_weight1(Xs,Ys,features)

media_pesi=pesi.mean(axis=1) #calcolo media valori per riga
media_pesi=media_pesi.sort_values( ascending=False) #metto valori in ordine decrescente
print("Le 50 feature piu importanti sono: ")
print(media_pesi.iloc[:50])
print("Elenco feature: ")
print(media_pesi.iloc[:50].index)
'''
'''
#Get prediction dataset 
X_test, Y_test = utl.get_ps_data_to_predict()   
#print(X_test.shape)
Y_pred = model.predict(X_test) 
results = model.evaluate(X_test, Y_test) 
utl.nn_evaluate(results) 

'''

