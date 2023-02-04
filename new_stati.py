import new_utl as utl 
import numpy as np 
from random import randint as r
import pandas as pd

# Examples with 5 features:
# Descrizione:
# ogni paziente ha 5 features 
# ogni feature vale 0 oppure 1
# dico che un paziente "ha una feature" se esiste una 
# sola feature uguale a uno, non importa quale.
# dico che un paziente "ha N feature" se il pazioente ha
# N features uguali a 1, no importa quali.
# Definisco una classificazione a tre stati 0,1,2:
# Un paziente è in stato
# 0 se ha alpiù una feature
# 1 se almeno 2 feature e alpiù 3 feature
# 2 se almeno 4 feature e alpiù 5 feature
# Lo stato 0 viene indicato con L ("Low" risk)
# Lo stato 1 viene indicato con M ("Moderate" risk)
# Lo stato 2 viene indicato con H ("High" risk)

# Modello a 3 stati con 1 singolo output:
# Input della rete: Xs, dove ogni elemento di Xs è un vettore di 5 features
# Output della rete (già addestrata): Ys, dove ogni elemento di Ys è un numero in [0,2]

# NOTA IMPORTANTE: il valore degli elementi di Y cambia a seconda
# che si stia facendo classificazione o regressione.
# Nella classificazione (default classif=True) l'output del modello
# è un numero intero, ovvero y in {0,1,2}.
# Nella regressione (classif=False) l'output del modello è un
# numero floating point, ad esempio 1,42. Ovvero
# ci viene restituita una predizione.

# Esempio:
# classificazione (precisa al 100%)
# Xs = [ [0,1,0,1,1], [1,0,0,0,0] ]
# Ys = [ 2, 0 ] 
# regressione (non precisa al 100%, ma molto precisa...)
# Xs = [ [0,1,0,1,1], [1,0,0,0,0] ]
# Ys = [ 1.95, 0.03 ] 


# Modello a 3 stati con 3 output:
# Input della rete: Xs, dove ogni elemento di Xs è un vettore di 5 features
# Output della rete (già addestrata): Ys, dove ogni elemento di Ys è un vettore di 3 elementi
# in [0,1]. 

# NOTA IMPORTANTE: il valore degli elementi di Y cambia a seconda
# che si stia facendo classificazione o regressione.
# Nella classificazione (default classif=True) l'output del modello
# è un vettore di 3 numeri interi in [0,1].
# Nella regressione (classif=False) l'output del modello è un
# vettore di 3 numeri floating point che sommano a 1.
# Questo vettore di 3 numeri definisce una probabilità di 
# appartenenza ad ogni classe (solo in caso di regressione).

# Esempio:
# classificazione (precisa al 100%)
# Xs = [ [0,1,0,1,1], [1,0,0,0,0], [1,1,1,0,1] ]
# Ys = [ [0,1,0], [1,0,0], [0,0,1] ] 
# regressione (non precisa al 100%)
# Xs = [ [0,1,0,1,1], [1,0,0,0,0], [1,1,1,0,1] ]
# Ys = [ [0.1,0.8,0.1], [0.9,0.07,0.03], [0,0,1.0] ]

# le procedure seguenti fanno vedere classificazione
# o regressione (settando classif=False) per i modelli
# descritti. Il parametro n_patients corrisponde alla 
# dimensione del dataset, ovvero la cardinalità di Xs.


# Three state problem with one output
def toy_risk_assessment_1_output(n_patients,classif=True):

    # risk 
    L = 0 # low-risk 
    M = 1
    H = 2

    # computing dummy features
    record = lambda l: np.array([r(0,1) for _ in np.arange(l)])
    Xs = np.array([record(5) for _ in np.arange(n_patients)])
    Ys = np.empty(n_patients)

    # classify
    for x,i in zip(Xs,np.arange(len(Xs))):
        risk = x.sum()
        if risk < 2: # 0,1 Low risk
            Ys[i] = L
        elif risk < 4: # 2,3 Moderate risk
            Ys[i] = M 
        else:
            assert risk > 3 and risk < 6 # 4,5 High risk
            Ys[i] = H
    
    model, weight = utl.rf_approach(Xs,Ys,[f'f{str(x)}' for x in np.arange(5)],classif=classif)

    # testing on new Xs:
    X_test = np.array([record(5) for _ in np.arange(20)])
    Y_pred = model.predict(X_test)

    X_test = pd.DataFrame(X_test)
    X_test['Y'] = Y_pred

    print(X_test)

    return 


# Three state problem with three outputs
def toy_risk_assessment_3_ouput(n_patients,classif=True):

    # risk 
    L = np.array([1,0,0])
    M = np.array([0,1,0])
    H = np.array([0,0,1])

    # computing dummy features
    record = lambda l: np.array([r(0,1) for _ in np.arange(l)])
    Xs = np.array([record(5) for _ in np.arange(n_patients)])
    Ys = np.empty((n_patients,3)) # n_patients x three states

    # classify
    for x,i in zip(Xs,np.arange(len(Xs))):
        risk = x.sum()
        if risk < 2: # 0,1 Low risk
            Ys[i] = L
        elif risk < 4: # 2,3 Moderate risk
            Ys[i] = M 
        else:
            assert risk > 3 and risk < 6 # 4,5 High risk
            Ys[i] = H
    
    model, weight = utl.rf_approach(Xs,Ys,[f'f{str(x)}' for x in np.arange(5)],classif=classif)

    # testing on new Xs:
    X_test = np.array([record(5) for _ in np.arange(20)])
    Y_pred = model.predict(X_test)

    X_test = pd.DataFrame(X_test,columns=[f'ft_{i}' for i in np.arange(5)])
    Y_pred = pd.DataFrame(Y_pred,columns=['low','moderate','high'])
    
    X_test = pd.concat((X_test,Y_pred),axis=1)

    print(X_test)

    return 

