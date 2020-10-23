#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:36:11 2019

@author: edouardduquesne
"""

import pandas as pd
from calcul_features import calcul_features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor

# chargement des labels
SBP_label = pd.read_csv("/Users/edouardduquesne/Desktop/Hinlab/SmallDataSet/SBP_label.csv", delimiter = ';', header = None)
DBP_label = pd.read_csv("/Users/edouardduquesne/Desktop/Hinlab/SmallDataSet/DBP_label.csv", delimiter = ';', header = None)

# chargement des fichiers segmentés
fichier_segmente = pd.read_csv("/Users/edouardduquesne/Desktop/Hinlab/SmallDataSet/segmente.csv", delimiter = ';', header = None)
# chargement des features liés à l'anthropometrie du sujet : age, poids, taille, sex
features_anthropo = pd.read_csv("/Users/edouardduquesne/Desktop/Hinlab/SmallDataSet/features_anthropo.csv", delimiter = ';')

BUFFER_SIZE = 1200

# on calcule tous les features liés au signal et on ajoute aussi les features anthropométriques au même tableau de features
features = calcul_features(fichier_segmente,features_anthropo,SBP_label,BUFFER_SIZE)

# permet de choisir quels features on souhaite garder pour l'entrainement du modèles
features = pd.concat([features.PPG_mean, features.PPG_min,features.PPG_max,features.PPG_amp,features.PPG_q_0_75, 
                                    features.PPG_q_0_25,features.PPG_0_cross,features.PPG_kurt,features.PPG_var,features.PPG_len, 
                                    features.PPG_S_len, features.PPG_D_len,features.PPG_int, features.PPG_S_int, features.PPG_D_int,
                                    features.PPG_len_S_tot,features.PPG_len_D_tot, features.PPG_len_D_S, 
                                    features.PPG_int_S_tot, features.PPG_int_D_tot, features.PPG_int_D_S,features.PPG_max_D, 
                                    features.PPG_S_High_0_10, features.PPG_S_High_0_25,features.PPG_S_High_0_33,features.PPG_S_High_0_50,
                                    features.PPG_S_High_0_66,features.PPG_S_High_0_75,
                                    features.PPG_D_High_0_10, features.PPG_D_High_0_25,features.PPG_D_High_0_33,features.PPG_D_High_0_50,
                                    features.PPG_D_High_0_66,features.PPG_D_High_0_75,
                                    features.sex, features.age, features.weight, features.height, features.bmi], axis =1)

X_train = features.values

# le signal est dejà normalisé mais il est important de normaliser/scaler les features entre eux
max_abs_scaler = preprocessing.MaxAbsScaler()
#min_max_scaler = preprocessing.MinMaxScaler()
X_train = max_abs_scaler.fit_transform(X_train)
#X_train = min_max_scaler.fit_transform(X_train)
#X_train = normalize(X_train, norm = 'max')
"""sc = StandardScaler()
X_train = sc.fit_transform(X_train)"""

# on divise les données en données d'entrainement et de test 
X_train, X_test, y_train, y_test = train_test_split(X_train, SBP_label, test_size=0.2, random_state=0)

# utiliser de gridsearchCV pour extraire les paramètres les plus appropriés
# intéressant aussi car cross validation auto

#random forest regressor
params_rfr = {
    "n_estimators": [5,10,15,20,25,30,35],
    "max_depth" : [10,15,20,25,30],
    "random_state" : [0]
}
rfr = RandomForestRegressor()
gsc_rfr = GridSearchCV(rfr, params_rfr, cv=7)

gsc_rfr.fit(X_train, np.ravel(y_train))
y_pred = gsc_rfr.best_estimator_.predict(X_test)

# decision tree regressor
"""dtr = DecisionTreeRegressor()
params_dtr = {
    "max_depth" : [10,15,20,25,30],
}
gsc_dtr = GridSearchCV(dtr, params_dtr, cv=7)
gsc_dtr.fit(X_train, np.ravel(y_train))
y_pred = gsc_dtr.best_estimator_.predict(X_test)"""

# support vector regressor
"""params_svr = {
    "kernel": ['linear'],
    "C" : [100],
    "gamma" : ['auto']
}
svr = SVR()
gsc_svr = GridSearchCV(svr, params_svr, cv=7)
gsc_svr.fit(X_train, np.ravel(y_train))
y_pred = gsc_svr.best_estimator_.predict(X_test)"""

# adaboost regressor
'''adb = AdaBoostRegressor(random_state=0, n_estimators=100)
adb.fit(X_train, y_train)
y_pred = adb.predict(X_test)'''

"""adb = AdaBoostRegressor(random_state=0, n_estimators=100)
adb.fit(X_train, y_train)
y_pred = adb.predict(X_test)"""

# linear regression
"""lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)"""

# decision tree regression
"""dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)"""

y_pred = y_pred.reshape((-1,1))
y_test = y_test.values

error_moy = np.mean(y_pred-y_test)
std = np.std(y_pred-y_test)

plt.figure()
plt.plot(y_test, y_pred, 'bo')
plt.title('SBP predite en fonction de SBP réelle')
plt.xlabel('SBP réelle')
plt.ylabel('SBP prédite')

plt.show()

print('Erreur moyenne',error_moy)
print('Ecart type',std)