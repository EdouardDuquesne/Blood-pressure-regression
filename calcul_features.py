# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:35:38 2019

@author: Edouard Duquesne
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import integrate
import matplotlib.pyplot as plt

def calcul_features(PPG,features_anthropo, Ytrain,BUFFER_SIZE):
    #calcul des features pour la prediction
    PPG = PPG.values
    feature = pd.DataFrame(index = Ytrain, columns = ['PPG_mean', 'PPG_min','PPG_max','PPG_std','PPG_amp','PPG_q_0_75', 
                                        'PPG_q_0_25','PPG_0_cross','PPG_kurt','PPG_skew','PPG_var','PPG_len', 'PPG_S_len', 'PPG_D_len',
                                        'PPG_int', 'PPG_S_int', 'PPG_D_int','PPG_len_S_tot','PPG_len_D_tot', 'PPG_len_D_S', 
                                        'PPG_int_S_tot', 'PPG_int_D_tot', 'PPG_int_D_S','PPG_max_D', 
                                        'PPG_S_High_0_10', 'PPG_S_High_0_25','PPG_S_High_0_33','PPG_S_High_0_50','PPG_S_High_0_66','PPG_S_High_0_75',
                                        'PPG_D_High_0_10', 'PPG_D_High_0_25','PPG_D_High_0_33','PPG_D_High_0_50','PPG_D_High_0_66','PPG_D_High_0_75',
                                        'sex', 'age', 'weight', 'height', 'bmi',
                                        'D_PPG_mean','D_PPG_min','D_PPG_max','D_PPG_std','D_PPG_amp','D_PPG_q_0_75', 
                                        'D_PPG_q_0_25','D_PPG_0_cross','D_PPG_kurt','D_PPG_skew'])
    
    PPG_mean = np.zeros(PPG.shape[0]) # moyenne signal
    PPG_min = np.zeros(PPG.shape[0]) # minimum signal
    PPG_max = np.zeros(PPG.shape[0]) # max signal
    PPG_std = np.zeros(PPG.shape[0]) # ecart type signal
    PPG_amp = np.zeros(PPG.shape[0]) # amplitude signal
    PPG_0_75 = np.zeros(PPG.shape[0]) # percentile 0.75 signal
    PPG_0_25 = np.zeros(PPG.shape[0]) # percentile 0.25 signal
    PPG_0_cross = np.zeros(PPG.shape[0]) # nombre d'intersection à 0
    PPG_kurt = np.zeros(PPG.shape[0]) # kurtosis signal
    PPG_skew = np.zeros(PPG.shape[0]) # skewness signal
    PPG_var = np.zeros(PPG.shape[0]) # variance signal
    PPG_len = np.zeros(PPG.shape[0]) # longueur du signal
    PPG_S_len = np.zeros(PPG.shape[0]) # longueur temps systolique
    PPG_D_len = np.zeros(PPG.shape[0]) # lo,gueur temps diastolique
    PPG_int = np.zeros(PPG.shape[0]) # integrale signal
    PPG_S_int = np.zeros(PPG.shape[0]) # integrale aire systolique
    PPG_D_int = np.zeros(PPG.shape[0]) # integrale aire diastolique
    PPG_len_S_tot = np.zeros(PPG.shape[0]) # rapppport temps systolique/temps total
    PPG_len_D_tot = np.zeros(PPG.shape[0]) # rapppport temps diastolique/temps total
    PPG_len_D_S = np.zeros(PPG.shape[0]) # rapppport temps diastolique/temps systolique
    PPG_int_S_tot = np.zeros(PPG.shape[0]) # rapppport integrale systolique/integrale total
    PPG_int_D_tot = np.zeros(PPG.shape[0]) # rapppport integrale diastolique/integrale total
    PPG_int_D_S = np.zeros(PPG.shape[0]) # rapppport integrale diastolique/integrale systolique
    PPG_max_D = np.zeros(PPG.shape[0]) # max de la dérivéé du signal
    PPG_S_High_0_10 = np.zeros(PPG.shape[0]) # abscisse à 0.10*max_abs côté gauche
    PPG_S_High_0_25 = np.zeros(PPG.shape[0]) # abscisse à 0.25*max_abs côté gauche
    PPG_S_High_0_33 = np.zeros(PPG.shape[0]) # abscisse à 0.33*max_abs côté gauche
    PPG_S_High_0_50 = np.zeros(PPG.shape[0]) # abscisse à 0.50*max_abs côté gauche
    PPG_S_High_0_66 = np.zeros(PPG.shape[0]) # abscisse à 0.66*max_abs côté gauche
    PPG_S_High_0_75 = np.zeros(PPG.shape[0]) # abscisse à 0.75*max_abs côté gauche
    
    PPG_D_High_0_10 = np.zeros(PPG.shape[0]) # abscisse à 0.10*max_abs côté droit
    PPG_D_High_0_25 = np.zeros(PPG.shape[0]) # abscisse à 0.25*max_abs côté droit
    PPG_D_High_0_33 = np.zeros(PPG.shape[0]) # abscisse à 0.33*max_abs côté droit
    PPG_D_High_0_50 = np.zeros(PPG.shape[0]) # abscisse à 0.50*max_abs côté droit
    PPG_D_High_0_66 = np.zeros(PPG.shape[0]) # abscisse à 0.66*max_abs côté droit
    PPG_D_High_0_75 = np.zeros(PPG.shape[0]) # abscisse à 0.75*max_abs côté droit
    
    D_PPG = np.zeros((PPG.shape[0],PPG.shape[1])) # creation du vecteur derivee du signal
    D_PPG_mean = np.zeros(PPG.shape[0]) # moyenne derivee signal
    D_PPG_min = np.zeros(PPG.shape[0]) # min derivee signal
    D_PPG_max = np.zeros(PPG.shape[0]) # max derivee signal 
    D_PPG_std = np.zeros(PPG.shape[0]) # ecart type derivee signal
    D_PPG_amp = np.zeros(PPG.shape[0]) # amplitude derivee
    D_PPG_0_75 = np.zeros(PPG.shape[0]) # percentile 0.75 derivee
    D_PPG_0_25 = np.zeros(PPG.shape[0]) # percentile 0.25 derivee
    D_PPG_0_cross = np.zeros(PPG.shape[0]) # intersections à 0 de la derivee
    D_PPG_kurt = np.zeros(PPG.shape[0]) # kurtosis de la derivee
    D_PPG_skew = np.zeros(PPG.shape[0]) # skewness de la derivee
    
    for i in range(PPG.shape[0]):
        cpt = 1
        # Buffer_size doit être égal au size_exp du fichier read.py
        # la fin du signal est composée de 0, on revient en arrière jusqu'à ce qu'il n'y est plus de 0 pour
        # connaitre la taille exacte du signal
        # une fois celle-ci connue on peut calculer les features sur le signal
        while PPG[i,BUFFER_SIZE - cpt] == PPG[i,BUFFER_SIZE-1]:
            cpt = cpt + 1
            
        """if i < 10:
            plt.figure()
            plt.plot(PPG[i,0:BUFFER_SIZE - cpt])
            plt.show()"""
            
        PPG_mean[i] =  np.mean(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_min[i] =  np.min(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_max[i] =  np.max(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_std[i] =  np.std(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_amp[i] =  np.max(PPG[i,0:BUFFER_SIZE - cpt]) - np.min(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_0_75[i] = np.quantile(PPG[i,0:BUFFER_SIZE - cpt], 0.75)
        PPG_0_25[i] = np.quantile(PPG[i,0:BUFFER_SIZE - cpt], 0.25)
        PPG_0_cross[i] = len(np.where(np.diff(np.sign(PPG[i,0:BUFFER_SIZE - cpt])))[0])
        PPG_kurt[i] = kurtosis(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_skew[i] =  skew(PPG[i,0:BUFFER_SIZE - cpt])
        
        if np.mean(PPG[i,0:BUFFER_SIZE - cpt]) == 0:
            PPG_var[i] =  float(np.std(PPG[i,0:BUFFER_SIZE - cpt]))
        else:
            PPG_var[i] =  float(np.std(PPG[i,0:BUFFER_SIZE - cpt])) / float(np.mean(PPG[i,0:BUFFER_SIZE - cpt]))
        
        PPG_len[i] = BUFFER_SIZE-cpt
        PPG_S_len[i] = np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_D_len[i] = BUFFER_SIZE - cpt - np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
        PPG_int[i] = np.trapz(PPG[i,0:BUFFER_SIZE - cpt], dx = 1/1000)
        PPG_S_int[i] = np.trapz(PPG[i,0:np.argmax(PPG[i,0:BUFFER_SIZE - cpt])], dx = 1/1000)
        PPG_D_int[i] = np.trapz(PPG[i,np.argmax(PPG[i,0:BUFFER_SIZE - cpt]):BUFFER_SIZE - cpt], dx = 1/1000)
        PPG_len_S_tot[i] = PPG_S_len[i]/PPG_len[i]
        PPG_len_D_tot[i] = PPG_D_len[i]/PPG_len[i]
        PPG_len_D_S[i] = PPG_D_len[i]/PPG_S_len[i]
        PPG_int_S_tot[i] = PPG_S_int[i]/PPG_int[i]
        PPG_int_D_tot[i] = PPG_D_int[i]/PPG_int[i]
        PPG_int_D_S[i] = PPG_D_int[i]/PPG_S_int[i]
        PPG_max_D[i] = np.max(np.gradient(PPG[i,0:BUFFER_SIZE - cpt], 1/1000))
        
        for k in range(np.argmax(PPG[i,0:BUFFER_SIZE - cpt])):
            if PPG[i,k] < 0.10 * PPG_max[i] and PPG[i,k+1] > 0.10 * PPG_max[i]:
                PPG_S_High_0_10[i] = np.argmax(PPG[i,0:BUFFER_SIZE - cpt]) - k
                
            if PPG[i,k] < 0.25 * PPG_max[i] and PPG[i,k+1] > 0.25 * PPG_max[i]:
                PPG_S_High_0_25[i] = np.argmax(PPG[i,0:BUFFER_SIZE - cpt]) - k
            
            if PPG[i,k] < 0.33 * PPG_max[i] and PPG[i,k+1] > 0.33 * PPG_max[i]:
                PPG_S_High_0_33[i] = np.argmax(PPG[i,0:BUFFER_SIZE - cpt]) - k
            
            if PPG[i,k] < 0.50 *PPG_max[i] and PPG[i,k+1] > 0.50 * PPG_max[i]:
                PPG_S_High_0_50[i] = np.argmax(PPG[i,0:BUFFER_SIZE - cpt]) - k
                
            if PPG[i,k] < 0.66 *PPG_max[i] and PPG[i,k+1] > 0.66 * PPG_max[i]:
                PPG_S_High_0_66[i] = np.argmax(PPG[i,0:BUFFER_SIZE - cpt]) - k
            
            if PPG[i,k] < 0.75 * PPG_max[i] and PPG[i,k+1] > 0.75 * PPG_max[i]:
                PPG_S_High_0_75[i] = np.argmax(PPG[i,0:BUFFER_SIZE - cpt]) - k
        
        
        for k in range(np.argmax(PPG[i,0:BUFFER_SIZE - cpt]),BUFFER_SIZE - cpt):
            if PPG[i,k] > 0.10 * PPG_max[i]and PPG[i,k+1] < 0.10 * PPG_max[i]:
                PPG_D_High_0_10[i] = k - np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
                
            if PPG[i,k] > 0.25 * PPG_max[i] and PPG[i,k+1] < 0.25 * PPG_max[i]:
                PPG_D_High_0_25[i] = k - np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
            
            if PPG[i,k] > 0.33 * PPG_max[i] and PPG[i,k+1] < 0.33 * PPG_max[i]:
                PPG_D_High_0_33[i] = k - np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
            
            if PPG[i,k] > 0.50 * PPG_max[i] and PPG[i,k+1] < 0.50 * PPG_max[i]:
                PPG_D_High_0_50[i] = k - np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
                
            if PPG[i,k] > 0.66 * PPG_max[i] and PPG[i,k+1] < 0.66 * PPG_max[i]:
                PPG_D_High_0_66[i] = k - np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
            
            if PPG[i,k] > 0.75 * PPG_max[i] and PPG[i,k+1] < 0.75 * PPG_max[i]:
                PPG_D_High_0_75[i] = k - np.argmax(PPG[i,0:BUFFER_SIZE - cpt])
                
        D_PPG[i,:] = np.gradient(PPG[i,:], 1/1000)
        """print(D_PPG)
        plt.figure()
        plt.plot(D_PPG)
        plt.show()"""
        
        D_PPG_mean[i] =  np.mean(D_PPG[i,0:BUFFER_SIZE - cpt])
        D_PPG_min[i] =  np.min(D_PPG[i,0:BUFFER_SIZE - cpt])
        D_PPG_max[i] =  np.max(D_PPG[i,0:BUFFER_SIZE - cpt])
        D_PPG_std[i] =  np.std(D_PPG[i,0:BUFFER_SIZE - cpt])
        D_PPG_amp[i] =  np.max(D_PPG[i,0:BUFFER_SIZE - cpt]) - np.min(PPG[i,0:BUFFER_SIZE - cpt])
        D_PPG_0_75[i] = np.quantile(D_PPG[i,0:BUFFER_SIZE - cpt], 0.75)
        D_PPG_0_25[i] = np.quantile(D_PPG[i,0:BUFFER_SIZE - cpt], 0.25)
        D_PPG_0_cross[i] = len(np.where(np.diff(np.sign(D_PPG[i,0:BUFFER_SIZE - cpt])))[0])
        D_PPG_kurt[i] = kurtosis(D_PPG[i,0:BUFFER_SIZE - cpt])
        D_PPG_skew[i] =  skew(D_PPG[i,0:BUFFER_SIZE - cpt])
        
    feature.PPG_mean = PPG_mean
    feature.PPG_min = PPG_min
    feature.PPG_max = PPG_max
    feature.PPG_std = PPG_std
    feature.PPG_amp = PPG_amp
    feature.PPG_q_0_75 = PPG_0_75
    feature.PPG_q_0_25 = PPG_0_25
    feature.PPG_0_cross = PPG_0_cross
    feature.PPG_kurt = PPG_kurt
    feature.PPG_skew = PPG_skew
    feature.PPG_var = PPG_var
    feature.PPG_len = PPG_len
    feature.PPG_S_len = PPG_S_len
    feature.PPG_D_len = PPG_D_len
    feature.PPG_int = PPG_int
    feature.PPG_S_int = PPG_S_int
    feature.PPG_D_int = PPG_D_int
    feature.PPG_len_S_tot = PPG_len_S_tot
    feature.PPG_len_D_tot = PPG_len_D_tot
    feature.PPG_len_D_S = PPG_len_D_S
    feature.PPG_int_S_tot = PPG_int_S_tot
    feature.PPG_int_D_tot = PPG_int_D_tot
    feature.PPG_int_D_S = PPG_int_D_S
    feature.PPG_max_D = PPG_max_D
    
    feature.PPG_S_High_0_10 = PPG_S_High_0_10
    feature.PPG_S_High_0_25 = PPG_S_High_0_25
    feature.PPG_S_High_0_33 = PPG_S_High_0_33
    feature.PPG_S_High_0_50 = PPG_S_High_0_50
    feature.PPG_S_High_0_66 = PPG_S_High_0_66
    feature.PPG_S_High_0_75 = PPG_S_High_0_75
    
    feature.PPG_D_High_0_10 = PPG_D_High_0_10
    feature.PPG_D_High_0_25 = PPG_D_High_0_25
    feature.PPG_D_High_0_33 = PPG_D_High_0_33
    feature.PPG_D_High_0_50 = PPG_D_High_0_50
    feature.PPG_D_High_0_66 = PPG_D_High_0_66
    feature.PPG_D_High_0_75 = PPG_D_High_0_75
    
    feature.D_PPG_mean = D_PPG_mean
    feature.D_PPG_min = D_PPG_min
    feature.D_PPG_max = D_PPG_max
    feature.D_PPG_std = D_PPG_std
    feature.D_PPG_amp = D_PPG_amp
    feature.D_PPG_q_0_75 = D_PPG_0_75
    feature.D_PPG_q_0_25 = D_PPG_0_25
    feature.D_PPG_0_cross = D_PPG_0_cross
    feature.D_PPG_kurt = D_PPG_kurt
    feature.D_PPG_skew = D_PPG_skew
    
    feature.sex = features_anthropo['feat_sex'].values
    feature.age = features_anthropo['feat_age'].values
    feature.weight = features_anthropo['feat_weight'].values
    feature.height = features_anthropo['feat_height'].values
    feature.bmi = features_anthropo['feat_bmi'].values
    
    ## fin calcul features
    
    return feature