#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:05:31 2022

@author: pertsevm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import datetime, os
import argparse

#######################################################
########## max and min length of CDR loops ############
max_CDR123_ab_len = 77 #72 + 5'_' signs
max_CDR3_ab_len = 47 #46 + '_' sign to join the loops
max_CDR_123b_len = 38 #36 + 2'_'signs
max_CDR_3b_len = 23 #max len of CDR3b


#make a dict with data input ('beta') and columns to take&max_len
model_input_type = {}
keys = ['beta', 'beta_VJ', 'albeta', 'albeta_VJ']
values = [(['CDR3b'], max_CDR_3b_len), (['CDR1b','CDR2b','CDR3b'], max_CDR_123b_len), \
          (['CDR3a', 'CDR3b'], max_CDR3_ab_len), (['CDR1a', 'CDR2a',  'CDR3a', 'CDR1b', 'CDR2b',  'CDR3b'], max_CDR123_ab_len)]
model_input_dict = dict(zip(keys, values))



######################################################################
################# Open TCR PAIRED repertoire files ###################

directory_TCR = "./Datasets/TCR_datasets/TCR_paired/"

train = directory_TCR + 'TCRab_Hom_Mus_train.csv'
val = directory_TCR + 'TCRab_Hom_Mus_val.csv'
test = directory_TCR + 'TCRab_Hom_Mus_test.csv'

test_ab_85 = directory_TCR + 'TCRab_test_ab_85.csv'
test_ab_80 = directory_TCR + 'TCRab_test_ab_80.csv'
test_ab_75 = directory_TCR + 'TCRab_test_ab_75.csv'
test_ab_70 = directory_TCR + 'TCRab_test_ab_70.csv'

test_b_85 = directory_TCR + 'TCRab_test_b_85.csv'
test_b_80 = directory_TCR + 'TCRab_test_b_80.csv'
test_b_75 = directory_TCR + 'TCRab_test_b_75.csv'
test_b_70 = directory_TCR + 'TCRab_test_b_70.csv'


#######################################################################
###################### Open TCRpMHC file NEW ##########################

directory_TCRpMHC = './Datasets/pMHC_TCR_dataset/TCRpMHC_new/train_data/'
TCRpMHC_new_data_90cdhit = directory_TCRpMHC + 'TCRpMHC_data_with_folds_90cdhit.csv'



#######################################################################
###################### Open TCRpMHC files #############################

directory_TCRpMHC = "./Datasets/pMHC_TCR_dataset/Combined_final_TCRpMHC_datasets/"

#Balanced datasets 
# with cd-hit
TCRpMHC_paired_VJ_95cdh = directory_TCRpMHC + 'Balanced_data_95_similarity/paired_VJ_balanced_95__cdhit_similarity.tsv'
TCRpMHC_CDR3b_VJ_95cdh = directory_TCRpMHC + 'Balanced_data_95_similarity/CDR3b_VJ_balanced_95__cdhit_similarity.tsv'
TCRpMHC_paired_95cdh = directory_TCRpMHC + 'Balanced_data_95_similarity/paired_balanced_95__cdhit_similarity.tsv'
TCRpMHC_CDR3b_95cdh = directory_TCRpMHC + 'Balanced_data_95_similarity/CDR3b_balanced_95__cdhit_similarity.tsv'

TCRpMHC_paired_VJ_90cdh = directory_TCRpMHC + 'Balanced_data_90_similarity/paired_VJ_balanced_90__cdhit_similarity.tsv'
TCRpMHC_CDR3b_VJ_90cdh = directory_TCRpMHC + 'Balanced_data_90_similarity/CDR3b_VJ_balanced_90__cdhit_similarity.tsv'
TCRpMHC_paired_90cdh = directory_TCRpMHC + 'Balanced_data_90_similarity/paired_balanced_90__cdhit_similarity.tsv'
TCRpMHC_CDR3b_90cdh = directory_TCRpMHC + 'Balanced_data_90_similarity/CDR3b_balanced_90__cdhit_similarity.tsv'

#randomly balanced
TCRpMHC_paired_VJ_rndm_bal = directory_TCRpMHC + 'Randomly_balanced_datasets/Paired_VJ_random_balanced_400.tsv'
TCRpMHC_CDR3b_VJ_rndm_bal = directory_TCRpMHC + 'Randomly_balanced_datasets/CDR3b_VJ_random_balanced_400.tsv'
TCRpMHC_paired_rndm_bal = directory_TCRpMHC + 'Randomly_balanced_datasets/Paired_random_balanced_400.tsv'
TCRpMHC_CDR3b_rndm_bal = directory_TCRpMHC + 'Randomly_balanced_datasets/CDR3b_random_balanced_400.tsv'

# not balanced at all
TCRpMHC_paired_VJ_unbalnce = directory_TCRpMHC + 'Unbalanced_datasets_eg_all_data/Paired_VJ_90_no_cross.tsv'
TCRpMHC_CDR3b_VJ_unbalnce = directory_TCRpMHC + 'Unbalanced_datasets_eg_all_data/CDR3b_VJ_90_no_cross.tsv'
TCRpMHC_paired_unbalnce = directory_TCRpMHC + 'Unbalanced_datasets_eg_all_data/Paired_90_no_cross.tsv'
TCRpMHC_CDR3b_unbalnce = directory_TCRpMHC + 'Unbalanced_datasets_eg_all_data/CDR3b_90_no_cross.tsv'

#######################################################################
#######################################################################




######################### OUTPUT DIRECTORY #################################
autoencoder_output_dir = "./Saved_models/CNN_AE/"
triplet_out_model= "./Saved_models/CNN_AE_TRIPLET/"
autoencoder_analysis_dir = "./Performance_files/CNN_AE_performance_and_latent_space/"
triplet_analysis_dir = "./Performance_files/Triplet_performance_and_latent_space/"

############################################################################
