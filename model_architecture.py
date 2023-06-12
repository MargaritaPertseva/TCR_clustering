#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:32:20 2022

@author: pertsevm
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

def Autoencoder_CNN(input_shape, embedding_units, n_filters_list):  #N_FILTERS = [128, 64, 16]
    #encoder
    encoder_input = keras.Input(shape = input_shape)
    conv_1 = layers.Conv1D(filters = n_filters_list[0], kernel_size = 2, activation="relu", padding="valid")(encoder_input)
    conv_2 = layers.Conv1D(filters = n_filters_list[1], kernel_size = 2, activation="relu", padding="valid")(conv_1)
    conv_3 = layers.Conv1D(filters = n_filters_list[2], kernel_size = 2, activation='relu', padding="valid")(conv_2)
    x_flat = layers.Flatten()(conv_3)
    encoded = layers.Dense(units = embedding_units, activation = 'relu', name = 'encoded')(x_flat)
    
    #decoder
    x_flat_dcdr = layers.Dense(units = x_flat.shape[1])(encoded)
    x_2d = layers.Reshape((conv_3.shape[1], conv_3.shape[2]))(x_flat_dcdr)
    conv_3_dcdr = layers.Conv1DTranspose(filters = n_filters_list[2], kernel_size = 2, activation='relu', padding="valid")(x_2d)
    conv_2_dcdr = layers.Conv1DTranspose(filters = n_filters_list[1], kernel_size = 2, activation="relu", padding="valid")(conv_3_dcdr)
    conv_1_dcdr = layers.Conv1DTranspose(filters = n_filters_list[0], kernel_size = 2, activation="relu", padding="valid")(conv_2_dcdr)
    decoded = layers.Conv1DTranspose(filters = 21, kernel_size = 1, activation="softmax", padding="valid")(conv_1_dcdr)
    
    #autoencoder
    autoencoder = Model(encoder_input, decoded)
    
    return autoencoder


def Autoencoder_Vanilla(input_shape, embedding_units, hidden_units):

    #encoder
    encoder_input = keras.Input(shape=input_shape)
    input_layer = layers.Dense(encoder_input.shape[1], activation='relu')(encoder_input)
    hid_1 = layers.Dense(hidden_units[0], activation='relu')(input_layer)
    hid_2 = layers.Dense(hidden_units[1], activation='relu')(hid_1)
    encoded = layers.Dense(embedding_units, activation='relu')(hid_2)
    
    #decoder
    hid_2_dcdr = layers.Dense(hidden_units[1], activation='relu')(encoded)
    hid_1_dcdr = layers.Dense(hidden_units[0], activation='relu')(hid_2_dcdr)
    decoded = layers.Dense(encoder_input.shape[1], activation='sigmoid')(hid_1_dcdr) #relu or softmax or sigmoid?
    #print (decoded.shape)
    
    #autoencoder
    autoencoder = Model(encoder_input, decoded)
    
    return autoencoder


def Triplet_CNN_naive(input_shape, embedding_units):
    
    ## triplet model architecture
    triplet_loss_model = tf.keras.Sequential(
        [tf.keras.Input(shape=input_shape),
         tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu'),
         tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu'),
         tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(embedding_units, activation=None), # No activation on final dense layer
         tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))] # L2 normalize embeddings
    )
    
    return triplet_loss_model


def Triplet_CNN_pretrained(input_shape, autoencoder_model, n_of_AE_layers, trainable, \
                embedding_units, n_filters):
    
    ## take N layers from trained CNN_AE model
    cnn_trained_layers = tf.keras.Sequential()
    for layer in autoencoder_model.layers[:n_of_AE_layers]: 
        cnn_trained_layers.add(layer)
        
    ## freeze the taken layers or let them to be trained
    cnn_trained_layers.trainable = trainable #True or False
    
    ## triplet model architecture
    triplet_loss_model = tf.keras.Sequential(
        [tf.keras.Input(shape=input_shape),
         cnn_trained_layers,
         tf.keras.layers.Conv1D(filters=n_filters, kernel_size=2, padding='same', activation='relu'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(embedding_units, activation=None), # No activation on final dense layer
         tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))] # L2 normalize embeddings
    )
    
    return triplet_loss_model

def Triplet_CNN_fully_pretrained(input_shape, autoencoder_model, trainable, \
                embedding_units):
    
    ## take N layers from trained CNN_AE model
    cnn_encoder = tf.keras.Sequential()
    for layer in autoencoder_model.layers[:6]: 
        cnn_encoder.add(layer)
        
    ## freeze the taken layers or let them to be trained
    cnn_encoder.trainable = trainable #True or False
    
    ## triplet model architecture
    triplet_loss_model = tf.keras.Sequential(
        [tf.keras.Input(shape=input_shape),
         cnn_encoder, #outputs CNN embeddings with 'relu' acitvation'
         tf.keras.layers.Dense(embedding_units, activation=None), # No activation on final dense layer
         tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))] # L2 normalize embeddings
    )
    
    return triplet_loss_model