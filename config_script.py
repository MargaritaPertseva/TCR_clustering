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

#######################################################################
###################### Open TCRpMHC file ##############################

directory_TCRpMHC = './Datasets/pMHC_TCR_dataset/train_data/'
TCRpMHC_90cdhit = directory_TCRpMHC + 'TCRpMHC_data_with_folds_90cdhit.csv'


######################################################################
################# Open TCR PAIRED CHAIN repertoire files #############

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

test_files_dict = {'beta': ['test_b_85', 'test_b_80', 'test_b_75', 'test_b_70'], \
                   'beta_VJ': ['test_b_85', 'test_b_80', 'test_b_75', 'test_b_70'], \
                   'albeta': ['test_ab_85', 'test_ab_80', 'test_ab_75', 'test_ab_70'], \
                   'albeta_VJ': ['test_ab_85', 'test_ab_80', 'test_ab_75', 'test_ab_70']}


######################### OUTPUT DIRECTORY #################################
autoencoder_output_dir = "./Saved_models/CNN_AE/"
triplet_out_model= "./Saved_models/CNN_AE_TRIPLET/"
autoencoder_analysis_dir = "./Performance_files/CNN_AE_performance_and_latent_space/"
triplet_analysis_dir = "./Performance_files/Triplet_performance_and_latent_space/"

############################################################################
