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

import os 
import argparse

#import other python files
import config_script
import model_architecture
import enc_decode

# parse the command line
if __name__ == "__main__":
    # initialize Argument Parser
    parser = argparse.ArgumentParser()
    
    # to add different arguments we want to parse
    parser.add_argument("--model_type", type=str) #cnn or vanilla
    parser.add_argument("--input_type", type=str) #beta, albeta, beta_VJ, albeta_VJ
    parser.add_argument("--embedding_space", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--do_analysis", type=bool)
    parser.add_argument("--results_folder", type=str) #to record it
    #parser.add_argument("--n_filters", type=list)
    #parser.add_argument("--n_hidden", type=list)
    
# to read the arguments from the command line
args = parser.parse_args()
MODEL_TYPE = args.model_type
INPUT_TYPE = args.input_type
EMBEDDING_SPACE = args.embedding_space
EPOCHS = args.epochs
MODEL_NAME = args.model_name
DO_ANALYSIS = args.do_analysis

RESULTS_FOLDER = args.results_folder

if MODEL_TYPE=='CNN':
    #N_FILTERS = args.n_filters_list
    N_FILTERS = [128, 64, 16]
if MODEL_TYPE=='vanilla':
   #N_HIDDEN = args.n_hidden
   N_HIDDEN = [256, 128]
    

#####################################################################################
############################ Open TCRs and encode them ##############################

# open the files and assign some constant variables
train = pd.read_csv(config_script.train, header = 0)
val = pd.read_csv(config_script.val, header = 0)
test = pd.read_csv(config_script.test, header = 0)

model_input_dict = config_script.model_input_dict
model_input_key = INPUT_TYPE
df_columns = model_input_dict[model_input_key][0] #get columns to encode
max_seq_len = model_input_dict[model_input_key][1] #get max length for padding
print ('df_columns: ', df_columns, 'max_seq_len: ', max_seq_len)

# make a dist of one-hot encoded amino acids for data encoding:
AA_list = list("ACDEFGHIKLMNPQRSTVWY_")
AA_one_hot_dict = {char : l for char, l in zip(AA_list, np.eye(len(AA_list), k=0))}

#encode the data
X_train = enc_decode.CDR_one_hot_encoding(train, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
X_val = enc_decode.CDR_one_hot_encoding(val, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
X_test = enc_decode.CDR_one_hot_encoding(test, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
print ('Seqs encoded!')
print ('X_train shape: ', X_train.shape, 'X_val shape: ', X_val.shape, 'X_test shape: ', X_test.shape)

######################################################################################
############################ Set up & train a model  #################################

train_tensor = tf.data.Dataset.from_tensor_slices((X_train, X_train))
val_tensor = tf.data.Dataset.from_tensor_slices((X_val, X_val))
test_tensor = tf.data.Dataset.from_tensor_slices((X_test, X_test))
train_tensor = train_tensor.shuffle(1024).batch(32)
val_tensor = val_tensor.shuffle(1024).batch(32)
test_tensor = test_tensor.shuffle(1024).batch(32)

tf.keras.backend.clear_session()
optimizer = tf.keras.optimizers.Adam(lr=0.0003)

if MODEL_TYPE == 'CNN':
    autoencoder_model = model_architecture.Autoencoder_CNN(X_train.shape[1:], \
                                        embedding_units = EMBEDDING_SPACE, \
                                            n_filters_list = N_FILTERS)
    autoencoder_model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    autoencoder_model.summary()
    
if MODEL_TYPE == 'vanilla':
    X_train = X_train.reshape(-1, len(AA_list) * max_seq_len)
    X_val = X_val.reshape(-1, len(AA_list) * max_seq_len)
    X_test = X_test.reshape(-1, len(AA_list) * max_seq_len)
    autoencoder_model = model_architecture.Autoencoder_Vanilla(X_train.shape[1:], \
                                                               embedding_units = EMBEDDING_SPACE, \
                                                                   hidden_units = N_HIDDEN)
    autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003), loss='binary_crossentropy') #binary_crossentropy & sigmoid;
    autoencoder_model.summary()
    
# add early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', min_delta = 0.002, patience = 3, \
                   verbose = 1, restore_best_weights = True)
# add tensorboard
#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

print ('Starting training...')
# train the model
history = autoencoder_model.fit(
    train_tensor,
    epochs=30, 
    validation_data=val_tensor, 
    callbacks = [early_stop])  #[tensorboard_callback, early_stop]

print ('Model trained!')
# save it
autoencoder_model.save(config_script.autoencoder_output_dir + MODEL_NAME)

####################################################################################################
######################### Save model embeddings for future analysis ################################

if MODEL_TYPE == 'vanilla':
    FLAT_ENC = True
    last_enc_layer = 5
if MODEL_TYPE == 'CNN': 
    FLAT_ENC = False
    last_enc_layer = 6

autoencoder_model = keras.models.load_model(config_script.utoencoder_output_dir + MODEL_NAME)
encoder = tf.keras.Sequential()
decoder = tf.keras.Sequential()
for layer in autoencoder_model.layers[:last_enc_layer]: encoder.add(layer)
for layer in autoencoder_model.layers[last_enc_layer:]: decoder.add(layer)

####################################################
# 1) predict the 96-dim embeddings and save them
embeddings_val = encoder.predict(X_val)
embeddings_test = encoder.predict(X_test)
test_emb_all = pd.DataFrame(embeddings_test)
val_emb_all = pd.DataFrame(embeddings_val)

test_emb_all.to_csv(config_script.autoencoder_analysis_dir + RESULTS_FOLDER + '/' + INPUT_TYPE + '_test_embeddings.csv', header = True, index = None)
val_emb_all.to_csv(config_script.autoencoder_analysis_dir + RESULTS_FOLDER + '/' + INPUT_TYPE + '_val_embeddings.csv', header = True, index = None)

####################################################
# 2) calculate UMAP and PCA embeddings & save them
import umap
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

u = umap.UMAP(n_components = 2, min_dist = 0.3, n_neighbors = 15, metric = 'euclidean')
pca = PCA(n_components = 2)

df_umap_test = pd.DataFrame(u.fit_transform(embeddings_test), columns = ['axis1','axis2'])
df_umap_test['dim_reduction'] = 'umap'
df_pca_test = pd.DataFrame(pca.fit_transform(embeddings_test), columns = ['axis1','axis2'])
df_pca_test['dim_reduction'] = 'pca'
df_test = pd.concat([df_umap_test, df_pca_test], ignore_index = True)

test_ = test[['CDR3a', 'CDR3b', 'Va', 'Vb', 'Len_CDR3a', 'Len_CDR3b', 'Len_CDR3ab', 'species']]
test_labels = pd.concat([test_, test_], ignore_index = True)
test_emb_umap_pca = pd.concat([df_test, test_labels], axis=1)

test_emb_umap_pca.to_csv(config_script.autoencoder_analysis_dir + RESULTS_FOLDER + '/' + INPUT_TYPE + '_UMAP_PCA_test_emb.csv', header = True, index = None)



###################################################################################################
######################### Analyse the model reconstruction accuracy ###############################

if DO_ANALYSIS:
    ####################################################
    # 1) to check the reconstruction accuracy on VALIDATION set
    predictions_val = autoencoder_model.predict(X_val)
    gen_val_acc = enc_decode.recnstr_acc_all(X_val, predictions_val, flat_enc = FLAT_ENC)
    acc_per_loop_val = enc_decode.recnstr_acc_loops(X_val, predictions_val, flat_enc = FLAT_ENC)
    df_val = pd.DataFrame({'CDR_loop':df_columns, 'Reconstr_acc':acc_per_loop_val, 'Data':'val'})
    df_val.loc[len(df_val.index)] = ['all_loops', gen_val_acc, 'val'] 
    
    ####################################################  
    # 2) to check the reconstruction accuracy on TEST set
    predictions_test = autoencoder_model.predict(X_test)
    gen_test_acc = enc_decode.recnstr_acc_all(X_test, predictions_test, flat_enc = FLAT_ENC)
    acc_per_loop_test = enc_decode.recnstr_acc_loops(X_test, predictions_test, flat_enc = FLAT_ENC)
    df_test = pd.DataFrame({'CDR_loop':df_columns, 'Reconstr_acc':acc_per_loop_test, 'Data':'test'})
    df_test.loc[len(df_test.index)] = ['all_loops', gen_test_acc, 'test'] 
    
    #################################################### 
    # 3) to check the reconstruction accuracy on increasingly dissimilar datasets
    if INPUT_TYPE == 'beta' or INPUT_TYPE == 'beta_VJ':
        test_85 = pd.read_csv(config_script.test_b_85, header = 0)
        test_80 = pd.read_csv(config_script.test_b_80, header = 0)
        test_75 = pd.read_csv(config_script.test_b_75, header = 0)
        test_70 = pd.read_csv(config_script.test_b_70, header = 0)
        
    if INPUT_TYPE == 'albeta' or INPUT_TYPE == 'albeta_VJ':
        test_85 = pd.read_csv(config_script.test_ab_85, header = 0)
        test_80 = pd.read_csv(config_script.test_ab_80, header = 0)
        test_75 = pd.read_csv(config_script.test_ab_75, header = 0)
        test_70 = pd.read_csv(config_script.test_ab_70, header = 0)

    x_test_85 = enc_decode.CDR_one_hot_encoding(test_85, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
    x_test_80 = enc_decode.CDR_one_hot_encoding(test_80, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
    x_test_75 = enc_decode.CDR_one_hot_encoding(test_75, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
    x_test_70 = enc_decode.CDR_one_hot_encoding(test_70, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
    
    if MODEL_TYPE == 'vanilla':
        x_test_85 = x_test_85.reshape(-1, len(AA_list) * max_seq_len)
        x_test_80 = x_test_80.reshape(-1, len(AA_list) * max_seq_len)
        x_test_75 = x_test_75.reshape(-1, len(AA_list) * max_seq_len)
        x_test_70 = x_test_70.reshape(-1, len(AA_list) * max_seq_len)
    
    ## to calculate the accuracy and save it
    x_test_enc = [x_test_85, x_test_80, x_test_75, x_test_70]
    names = ['test_85', 'test_80', 'test_75', 'test_70']
    test_cdhit = pd.DataFrame()
    for i in range(len(x_test_enc)):
        predictions_ = autoencoder_model.predict(x_test_enc[i])
        acc_all_cdhit = enc_decode.recnstr_acc_all(x_test_enc[i], predictions_, flat_enc = FLAT_ENC)
        acc_per_loop_cdhit = enc_decode.recnstr_acc_loops(x_test_enc[i], predictions_, flat_enc = FLAT_ENC)
        df_ = pd.DataFrame({'CDR_loop':df_columns, 'Reconstr_acc':acc_per_loop_cdhit, 'Data':names[i]})
        df_.loc[len(df_.index)] = ['all_loops', acc_all_cdhit, names[i]]
        test_cdhit = test_cdhit.append(df_, ignore_index = True)

    reconstr_accuracy = pd.concat([df_val, df_test, test_cdhit], ignore_index = True)
    reconstr_accuracy['Model'] = MODEL_TYPE + '_' + INPUT_TYPE
    reconstr_accuracy.to_csv(config_script.autoencoder_analysis_dir + RESULTS_FOLDER + '/' + INPUT_TYPE + '_reconstr_perf.csv', header = True, index = False)
    
    
    

