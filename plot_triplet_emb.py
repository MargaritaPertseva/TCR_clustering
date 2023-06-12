#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:39:47 2022

@author: pertsevm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import umap
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

import os 
import argparse

#import other python files
import config_script
import model_architecture
import enc_decode

# parse the command line
if __name__ == "__main__":
    # initialize Argument Parser
    parser = argparse.ArgumentParser()    #help="Debug", nargs='?', type=int, const=1, default=7
    
    # we can add any different arguments we want to parse
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

    
# we then read the arguments from the comand line
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
TRIPLET_MODE = args.triplet_mode #'naive' #naive or pretrained
LOSS = args.triplet_loss #'semihard' or 'hard'
N_AE_LAYERS = args.n_trained_layers
TRAINABLE_LAYERS =  args.trainable_layers
N_FILTERS = args.n_filters
PLOT_EMB = args.plot_embedding

if TRAINABLE_LAYERS  == 'False':
    TRAINABLE_LAYERS = False
if TRAINABLE_LAYERS  == 'True':
    TRAINABLE_LAYERS  = True

#####################################################################################
########################## Calculate UMAP PCA #####################################  

def umap_pca_df(results_array, y_list, random_seed):
    
    import umap
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import StandardScaler
    no_of_components = 2
    pca = PCA(n_components=2, random_state=random_seed)
    u = umap.UMAP(n_components = 2, random_state=random_seed, min_dist = 0.3, n_neighbors = 15, metric = 'euclidean')
    
    
    df_umap = pd.DataFrame(u.fit_transform(results_array), columns = ['axis1','axis2'])
    df_umap['dim_reduction'] = 'umap'
    df_pca = pd.DataFrame(pca.fit_transform(results_array), columns = ['axis1','axis2'])
    df_pca['dim_reduction'] = 'pca'
    
    df_umap_pca = pd.concat([df_umap, df_pca], ignore_index = True)
    df_umap_pca['Epitope'] = y_list + y_list
    
    return df_umap_pca

#####################################################################################
########################## To plot latent space #####################################  
palette_22 = ['maroon','red', 'brown', 'orange', 'olive', \
            'lime', 'yellow', 'green', 'cyan', 'blue', \
            'teal', 'navy', 'pink', 'purple', 'lavender', \
            'magenta', 'peachpuff', 'beige', 'mint', 'black', \
            'white', 'grey']

palette_31 = ['#201923', '#ffffff', '#fcff5d','#7dfc00','#946aa2', \
           '#5d4c86','#0ec434','#228c68','#8ad8e8','#235b54', \
           '#29bdab','#3998f5','#37294f','#277da7','#3750db', \
           '#f22020','#991919','#ffcba5','#e68f66','#c56133', \
           '#96341c','#632819','#ffc413','#f47a22', '#2f2aa0', \
           '#b732cc','#772b9d','#f07cab','#d30b94','#edeff3', \
           '#c3a5b4']

color_dict = dict({'Africa':'brown',
                  'Asia':'green',
                  'Europe': 'orange',
                  'Oceania': 'red',
                   'Americas': 'dodgerblue'})

def plot_umap_pca_latent(df, hue_column, color_list, fig_title, legend_title, node_size, save, figname):
    
    # 1) set up sns style
    import seaborn as sns
    sns.set(font_scale = 2)
    sns.set_style("whitegrid")
    
    # 2) plot things
    g = sns.FacetGrid(df, col = 'dim_reduction', \
                  hue = hue_column, palette = color_list, sharex=False, sharey=False)  #palette=color_dict,
    g.map(sns.scatterplot, "axis1", "axis2", alpha=1, \
          s=node_size, edgecolor = 'darkslategrey', linewidth = 0.8) #linewidth = 0; or edgecolor = None, to remove white border
    
    g.fig.set_figwidth(32)
    g.fig.set_figheight(16)
    g.set_titles(col_template= fig_title, fontweight='bold', size=25)
    
    # set axis labels
    g.axes[0,0].set_xlabel('UMAP_1')
    g.axes[0,0].set_ylabel('UMAP_2')
    g.axes[0,1].set_xlabel('PCA_1')
    g.axes[0,1].set_ylabel('PCA_2')
    #g.set_yticklabels(g.get_yticks(), size = 15)
    
    # 3) add legend and adjust the size
    g.add_legend(title = legend_title)           #(legend_data=None, title=None, label_order=None)
    plt.setp(g._legend.get_title(), fontsize=25, fontweight='bold') # change the size of legend title
    plt.setp(g._legend.get_texts(), fontsize=20) # change the size of legend text
    
    # 4) save figure
    if save:
        g.savefig(config_script.triplet_analysis_dir + figname, \
                  dpi=500, bbox_inches='tight', pad_inches=0.5)
        
    return g



########################################################################################
############################ Open TCRpMHC and encode them ##############################

# open the files and assign some constant variables
TCRpMHC = pd.read_csv(config_script.TCRpMHC_new_data_90cdhit, header = 0)

model_input_dict = config_script.model_input_dict
model_input_key = INPUT_TYPE
df_columns = model_input_dict[model_input_key][0] #get columns to encode
max_seq_len = model_input_dict[model_input_key][1] #get max length for padding
print ('df_columns: ', df_columns, 'max_seq_len: ', max_seq_len)

# make a dist of one-hot encoded amino acids for data encoding:
AA_list = list("ACDEFGHIKLMNPQRSTVWY_")
AA_one_hot_dict = {char : l for char, l in zip(AA_list, np.eye(len(AA_list), k=0))}


#select train, val, test data  ###CHANGE THIS
train = TCRpMHC[TCRpMHC['part'].isin([0, 1, 2, 3, 4])]
val_test = TCRpMHC[TCRpMHC['part'].isin([5, 6])]
#test = TCRpMHC[TCRpMHC['part']==6]

#encode the data
X_train = enc_decode.CDR_one_hot_encoding(train, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
X_val_test = enc_decode.CDR_one_hot_encoding(val_test, df_columns, AA_list, AA_one_hot_dict, max_seq_len)

#save the Epi label for plotting
y_train_epi = train['Epitope'].tolist()
y_val_test_epi = val_test['Epitope'].tolist()

print ('X_train: ', X_train.shape, 'X_val_test: ',  X_val_test.shape)


########################################################################################
############################ Open triplet_model ##############################

# create a new model folder if there is none
if TRIPLET_MODE == 'naive':
    TRIPLET_NAME = f'triplet_{TRIPLET_MODE}_{INPUT_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence'
if TRIPLET_MODE == 'fully_pretrained':
    TRIPLET_NAME = f'triplet_{TRIPLET_MODE}_{INPUT_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence_trainable{TRAINABLE_LAYERS}_'
if TRIPLET_MODE == 'pretrained':
    TRIPLET_NAME = f'triplet_{TRIPLET_MODE}_{INPUT_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence_trainable{TRAINABLE_LAYERS}_Nfilters{N_FILTERS}'
    
# load the model
triplet = keras.models.load_model(config_script.triplet_out_model + TRIPLET_NAME)

results_val_test = triplet(X_val_test)
results_train = triplet(X_train)

umap_pca_val_test = umap_pca_df(results_val_test, y_val_test_epi, random_seed = 42)
plot_umap_pca_latent(umap_pca_val_test, 'Epitope', color_list = palette_31, \
                        fig_title = f'Epitope distribution by {TRIPLET_MODE} model with {INPUT_TYPE} triplet {EMBEDDING_SPACE}embedding ', \
                        legend_title = 'Epitope', node_size = 70, \
                        save = True, figname = f'Epitope_valtest_{TRIPLET_NAME}.jpg')

umap_pca_train = umap_pca_df(results_train, y_train_epi, random_seed = 42)
plot_umap_pca_latent(umap_pca_train, 'Epitope', color_list = palette_31, \
                        fig_title = f'Epitope distribution by {TRIPLET_MODE} model with {INPUT_TYPE} triplet {EMBEDDING_SPACE}embedding ', \
                        legend_title = 'Epitope', node_size = 70, \
                        save = True, figname = f'Epitope_train_{TRIPLET_NAME}.jpg')


    

