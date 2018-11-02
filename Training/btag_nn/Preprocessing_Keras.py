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


def transform_for_Keras(nb_classes):
  from keras.utils import np_utils

  FILE_PATH = '../Preprocessing/output/'  # '/eos/user/e/evillhau/new_double_b_tagger_ines/double-tagger-fun/Preprocessing/output_folder/'
  f=h5py.File(FILE_PATH + 'prepared_sample_v2.h5', 'r')
  wf=h5py.File(FILE_PATH + 'Weight_0.h5','r')

  X_train= f['X_train'][:]
  X_val= f['X_val'][:]
  X_test= f['X_test'][:]
  X_weights_train= f['X_weights_train'][:]
  X_weights_val=   f['X_weights_val'][:]
  X_weights_test=  f['X_weights_test'][:]

  Y_train=f['Y_train'][:]
  Y_val=  f['Y_val'][:]
  Y_test= f['Y_test'][:]

  arr_jet_pt_train=f['arr_jet_pt_train'][:]
  arr_jet_pt_val=  f['arr_jet_pt_val'][:]
  arr_jet_pt_test= f['arr_jet_pt_test'][:]
  arr_baseline_tagger_train=f['arr_baseline_tagger_train'][:]
  arr_baseline_tagger_val=  f['arr_baseline_tagger_val'][:]
  arr_baseline_tagger_test= f['arr_baseline_tagger_test'][:]

  # transforms label entries to int32 and array to binary class matrix as required for categorical_crossentropy:
  Y_train = np_utils.to_categorical(Y_train.astype(int), nb_classes)
  Y_test = np_utils.to_categorical(Y_test.astype(int), nb_classes)
  Y_val = np_utils.to_categorical(Y_val.astype(int), nb_classes)

  print ("Useful printouts:")
  print(X_weights_train)
  print(Y_train)
  print( arr_jet_pt_train )

  return X_train, Y_train,  X_test, Y_test,X_val, Y_val, X_weights_train, X_weights_test, X_weights_val, arr_baseline_tagger_train, arr_jet_pt_train, arr_baseline_tagger_test, arr_jet_pt_test, arr_baseline_tagger_val, arr_jet_pt_val


