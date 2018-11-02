'''
    Script to train NN for jet flavour identification purpose (b, c and light (and tau) jets): training and evaluation with Keras
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python btagging_nn.py
    Switch Keras backend using: KERAS_BACKEND=tensorflow python btagging_nn.py
'''
import os
import numpy as np
import pandas as pd
import json
import h5py


def transform_for_Keras (nb_classes):
  from keras.utils import np_utils

  FILE_PATH = '../Preprocessing/output/'

  with h5py.File(FILE_PATH + 'prepared_sample_v2.h5', 'r') as f:

    X       = f['X'][:]
    Y       = f['Y'][:]
    W_train = f['W_train'][:]    
    W_test  = f['W_test'] [:]    

    train = f['train'][:]
    test  = f['test'] [:]
    val   = f['val']  [:]
    
    arr_jet_pt   = f['arr_jet_pt']  [:]
    arr_jet_mass = f['arr_jet_mass'][:]
    arr_jet_eta  = f['arr_jet_eta'] [:]
    arr_baseline_tagger = f['arr_baseline_tagger'][:]
    
    # transforms label entries to int32 and array to binary class matrix as required for categorical_crossentropy:
    Y = np_utils.to_categorical(Y.astype(int), nb_classes)
    pass
  
  return X, Y, W_train, W_test, train, test, val, arr_baseline_tagger, arr_jet_pt, arr_jet_mass, arr_jet_eta


