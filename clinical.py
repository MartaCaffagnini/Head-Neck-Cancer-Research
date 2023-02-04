# NN WITH CLINICAL DATA. PREVISIONE A 2 STATI
import utility_nn as utl 
import pandas as pd
import numpy as np 

'''
Considerazioni importante:

I panzienti considerati nel dataset RS (restrospettivo),
utilizzato come input, sono 1086. Entro 10 anni dalla
data di diagnosi ne sono rimasti vivi 593 che è il numero
di persone vive nel dataset. Quindi è inutile chiedersi
quante persone sono morte dopo un periodo superiore 
a 10 anni pochè quell'informazione non è presente. 
Quindi il parametro 'alive_after' superiore a 10*365
non cambia i risultati di training.

Se però si considera un periodo inferiore ai 10 anni, 
allora il numero di pazienti acora vivi sale. Ad esempio
nel seguente codice si può calcolare il numero di persone
vive dopo 5 anni.

>>> rs_full = utl.get_rs_full_dataframe()
>>> Ys = utl.compute_alive_after(rs_full,'clinical_Date_of_first_Diagnosis','follow_Date_of_Death',5*365)
Number of people alive after 1825 days: 649 out of 1086
Maximum number of alive people in dataset: 593 out of 1086

Il risultato è 649. Ys è un vettore tale che Ys[i]==1 
se e solo se il paziente i-esimo (ogni riga di rs_full
è una cartella clinica del paziente i) è ancora vivo dopo
5 anni.

>>> Ys[1]
1
>>> Ys
array([1, 1, 0, ..., 0, 1, 1])
>>> Ys[1085]
1
>>> Ys[1083]
0
>>> Ys.sum() # numero di pazienti vivi dopo 5 anni
649

E' importante notare che cambiando il parametro 'alive_after'
cambia Ys e quindi cambia anche il modello ottenuto
con le random forest. In particolare il modello ottenuto
dovrebbe essere utilizzato per predirre la probabilità
di sopravvivienza di un paziente (o un set di pazienti)
dopo un periodo di 'alive_after' giorni.

Il dataset PS (prognostico) contiene in totale 442 righe
processabili. Le righe processabili vengono definite
(in questo test specifico) come i pazienti per i quali 
è nota la data di diagnosi. In PS, ci sono 336 pazienti
vivi. Riducendo un po' alla volta il parametro alive_after
si scopre come le restanti 106 persone sono decedute
entro 4 anni.  

>>> utl.compute_alive_after(ps,alive_after=1*365)
Number of people alive after 365 days: 405 out of 442
Maximum number of alive people in dataset: 336 out of 442
...
>>> utl.compute_alive_after(ps,alive_after=2*365)
Number of people alive after 730 days: 357 out of 442
Maximum number of alive people in dataset: 336 out of 442
...
>>> utl.compute_alive_after(ps,alive_after=3*365)
Number of people alive after 1095 days: 337 out of 442
Maximum number of alive people in dataset: 336 out of 442
...
>>> utl.compute_alive_after(ps,alive_after=4*365)
Number of people alive after 1460 days: 336 out of 442
Maximum number of alive people in dataset: 336 out of 442
...



'''
'''
# get training dataset
Xs, Ys, features = utl.get_rs_training_data(alive_after=5*365)
# Train on Xs,Ys
model, results = utl.nn_approach(Xs,Ys,features)
model_rf, weight, result_rf = utl.rf_approach(Xs,Ys,features)


acc = np.array(results[1])
acc_rf=np.array(result_rf)
for i in range(1,99):
 model, results= utl.nn_approach(Xs,Ys,features)
 acc =np.append( acc ,results[1])
 model_rf, weight, result_rf = utl.rf_approach(Xs,Ys,features)
 acc_rf =np.append( acc_rf ,result_rf)
 

acc_finale = np.average(acc)
av_acc_rf=np.average(acc_rf)
print(f"NN Average Accuracy Clinical Data: {acc_finale}")
print(f"RF Average Accuracy Clinical Data: {av_acc_rf}")
'''
#CON SOLO DATI AOP
# get training dataset
Xs, Ys, features = utl.get_rs_training_data_AOP(alive_after=5*365)
# Train on Xs,Ys
model, results = utl.nn_approach(Xs,Ys,features)
model_rf, weight, result_rf = utl.rf_approach(Xs,Ys,features)


acc = np.array(results[1])
acc_rf=np.array(result_rf)
for i in range(1,99):
 model, results= utl.nn_approach(Xs,Ys,features)
 acc =np.append( acc ,results[1])
 model_rf, weight, result_rf = utl.rf_approach(Xs,Ys,features)
 acc_rf =np.append( acc_rf ,result_rf)
 

acc_finale = np.average(acc)
av_acc_rf=np.average(acc_rf)
print(f"NN Average Accuracy Clinical Data: {acc_finale}")
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

