#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:39:47 2022

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
import tensorflow as tf
import tensorflow_addons as tfa
#import tensorflow_datasets as tfds


import csv
import os 
import argparse

#import other python files
import config_script
import model_architecture
import enc_decode
import plot_latent_space

# parse the command line
if __name__ == "__main__":
    # initialize Argument Parser
    parser = argparse.ArgumentParser()    #help="Debug", nargs='?', type=int, const=1, default=7
    
    # to add any different arguments we want to parse
    parser.add_argument("--model_type", help="CNN or vanilla", type=str, default='CNN') ## CNN or vanilla
    parser.add_argument("--input_type", help="beta, albeta, beta_VJ, albeta_VJ", type=str) ## beta, beta_VJ, albeta, albeta_VJ
    parser.add_argument("--embedding_space", type=int) ## embedding dimension for TCR
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--encoder_name", type=str)
    parser.add_argument("--patience", type = int, default = 15)
    #triplet hyperparameters
    parser.add_argument("--triplet_mode", help="naive, pretrained or fully_pretrained", type=str, default='pretrained')
    parser.add_argument("--triplet_loss", help="hard or semihard", type=str, default='semihard')
    parser.add_argument("--n_trained_layers", help = 'n of autoencoder layers to transfer', type=int, default = 3) #3, 4, 5
    parser.add_argument("--trainable_layers", help = 'allow to train transferred layers or not', type=str)
    parser.add_argument("--n_filters", help = 'n_filters for partly pretrained model', type=int, default = 16)
    parser.add_argument("--plot_embedding", type = bool, default = False)

    
# to read the arguments from the command line
args = parser.parse_args()
MODEL_TYPE = args.model_type
INPUT_TYPE = args.input_type
EMBEDDING_SPACE = args.embedding_space
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LRATE = args.learning_rate
PATIENCE = args.patience
ENCODER_NAME = args.encoder_name # ENCODER_NAME = 'cnn_newd_albeta_48_dim'

#triplet hyperparameters
TRIPLET_MODE = args.triplet_mode #naive or pretrained
LOSS = args.triplet_loss #'semihard' or 'hard'
N_AE_LAYERS = args.n_trained_layers
TRAINABLE_LAYERS =  args.trainable_layers
N_FILTERS = args.n_filters
PLOT_EMB = args.plot_embedding

print ('TRAINABLE_LAYERS : ', TRAINABLE_LAYERS)
if TRAINABLE_LAYERS  == 'False':
    TRAINABLE_LAYERS = False
if TRAINABLE_LAYERS  == 'True':
    TRAINABLE_LAYERS  = True
    
    
print ('TRAINABLE_LAYERS : ', TRAINABLE_LAYERS )
    
########################################################################################
############################ Open TCRpMHC and encode them ##############################

# to open the files and assign some constant variables
TCRpMHC = pd.read_csv(config_script.TCRpMHC_90cdhit, header = 0)

model_input_dict = config_script.model_input_dict
model_input_key = INPUT_TYPE
df_columns = model_input_dict[model_input_key][0] #get columns to encode
max_seq_len = model_input_dict[model_input_key][1] #get max length for padding
print ('df_columns: ', df_columns, 'max_seq_len: ', max_seq_len)

# to create a dict of one-hot encoded amino acids for TCR encoding
AA_list = list("ACDEFGHIKLMNPQRSTVWY_")
AA_one_hot_dict = {char : l for char, l in zip(AA_list, np.eye(len(AA_list), k=0))}

#select train, val, test data  ###CHANGE THIS
train = TCRpMHC[TCRpMHC['part'].isin([0, 1, 2, 3, 4])]
val = TCRpMHC[TCRpMHC['part']==5]
test = TCRpMHC[TCRpMHC['part']==6]

#encode the data
X_train = enc_decode.CDR_one_hot_encoding(train, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
X_val = enc_decode.CDR_one_hot_encoding(val, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
X_test = enc_decode.CDR_one_hot_encoding(test, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
X_all_data = enc_decode.CDR_one_hot_encoding(TCRpMHC, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
print ('Seqs encoded!')

y_train = train['label'].values
y_val = val['label'].values
y_test = test['label'].values

#save the Epi label for plotting
y_train_epi = train['Epitope'].tolist()
y_val_epi = val['Epitope'].tolist()
y_test_epi = test['Epitope'].tolist()

print ('X_train: ', X_train.shape,'X_val: ',  X_val.shape, 'X_test: ', X_test.shape)

#turn array into tensors
train_triplet = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_triplet = train_triplet.shuffle(1024).batch(BATCH_SIZE)
val_triplet = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_triplet = val_triplet.batch(BATCH_SIZE)


######################################################################################
############################ Set up & train a model  #################################

if LOSS == 'hard':
    loss = tfa.losses.TripletHardLoss()
if LOSS == 'semihard':
    loss = tfa.losses.TripletSemiHardLoss()


tf.keras.backend.clear_session()
if TRIPLET_MODE == 'naive':
    triplet = model_architecture.Triplet_CNN_naive(X_train.shape[1:], embedding_units = EMBEDDING_SPACE)
    
if TRIPLET_MODE == 'pretrained':
    autoencoder = keras.models.load_model(config_script.autoencoder_output_dir + ENCODER_NAME)
    triplet = model_architecture.Triplet_CNN_pretrained(X_train.shape[1:], autoencoder, n_of_AE_layers = N_AE_LAYERS, trainable = TRAINABLE_LAYERS , embedding_units = EMBEDDING_SPACE, n_filters = N_FILTERS)
    
if TRIPLET_MODE == 'fully_pretrained':
    autoencoder = keras.models.load_model(config_script.autoencoder_output_dir + ENCODER_NAME)
    triplet = model_architecture.Triplet_CNN_fully_pretrained(X_train.shape[1:], autoencoder, trainable = TRAINABLE_LAYERS , embedding_units = EMBEDDING_SPACE)
    
triplet.compile(optimizer=tf.keras.optimizers.Adam(LRATE), loss=loss) #tfa.losses.TripletSemiHardLoss()
triplet.summary()

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience = PATIENCE, \
                   verbose = 1, restore_best_weights = True)

print ('Starting training...')
history = triplet.fit(
    train_triplet,
    epochs=EPOCHS, 
    validation_data = val_triplet, 
    callbacks = [early_stop])

print ('Model trained!')

# Extract the best loss value (from early stopping) 
loss_hist = history.history['val_loss']
best_val_loss = np.min(loss_hist)
best_epochs = np.argmin(loss_hist) + 1



print ('TRAINABLE_LAYERS: ', TRAINABLE_LAYERS)
# create a new model folder if there is none
if TRIPLET_MODE == 'naive':
    TRIPLET_NAME = f'triplet_{TRIPLET_MODE}_{INPUT_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence'
if TRIPLET_MODE == 'fully_pretrained':
    TRIPLET_NAME = f'triplet_{TRIPLET_MODE}_{INPUT_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence_trainable{TRAINABLE_LAYERS}_'
if TRIPLET_MODE == 'pretrained':
    TRIPLET_NAME = f'triplet_{TRIPLET_MODE}_{INPUT_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence_trainable{TRAINABLE_LAYERS}_Nfilters{N_FILTERS}'
    
modelpath = config_script.triplet_out_model + TRIPLET_NAME
if not os.path.exists(modelpath):
    os.makedirs(modelpath)
# save the model
triplet.save(config_script.triplet_out_model + TRIPLET_NAME)


with open(config_script.triplet_analysis_dir + 'Triplet_Val_Loss.csv', 'a') as file:
    writer = csv.writer(file)
    data = [TRIPLET_NAME, best_val_loss, best_epochs]
    writer.writerow(data)
    
#print ('\n')
#print ('Model name: ', TRIPLET_NAME)
#print ('best loss: ', best_val_loss, 'best_epochs: ', best_epochs)
#print ('Triplet model saved!')

########################################################################################
############################## Plot model embeddings ###################################  
if PLOT_EMB == True:
    
    results_val = triplet(X_val)
    results_test = triplet(X_test)
    results_train = triplet(X_train)

    umap_pca_test = plot_latent_space.umap_pca_df(results_test, y_test_epi, random_seed = 42)
    plot_latent_space.plot_umap_pca_latent(umap_pca_test, 'Epitope', color_list = plot_latent_space.palette_31, \
                         fig_title = f'Epitope distribution by {TRIPLET_MODE} model with {INPUT_TYPE} triplet {EMBEDDING_SPACE}embedding ', \
                         legend_title = 'Epitope', node_size = 70, \
                         save = True, figname = f'Epitope_{TRIPLET_NAME}.jpg')








