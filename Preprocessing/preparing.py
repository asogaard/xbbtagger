'''
This the *third* script in the data preparation pipeline for the X->bb tagger
   preprocessing.py
   reweighting.py
 > preparing.py
'''

# Import(s)
import h5py
import numpy as np

import utilities.variable_info as variable_info
from utilities.common import *


def save_to_hdf5 (filename, datasets):
  """
  ...
  """
  with h5py.File(filename, 'w') as h5f:
    for name, data in datasets.iteritems():
      h5f.create_dataset(name, data=data)
      pass
    pass
  return


# Main function definition
def main ():

  # Parse command-line arguments
  args = parse_args()

  # Read data from pre-processed HDF5 file
  with h5py.File('{}/output_Preprocessed{}.h5'.format(args.output, args.nametag), 'r') as h5f:
      bbjets = h5f['arr_processed_bbjets'][:]
      dijets = h5f['arr_processed_dijets'][:]
      if args.ttbar:
          ttbar = h5f['arr_processed_ttbar'][:]
          pass
      pass

  # Read training weights from HDF5 file
  with h5py.File('{}/Weight_0{}.h5'.format(args.output, args.nametag), 'r') as wf:
      bb_weight_train = wf['bb_vs_bb_weights'][:]
      print("this is the shape of bb_weight_train", bb_weight_train.shape)
      di_weight_train = wf['dijet_vs_bb_weights'][:]
      #print(di_weight_train)
      print("this is the shape of di_weight_train", di_weight_train.shape)
      if args.ttbar:
          tt_weight_train = wf['ttbar_vs_bb_weights'][:]
          pass
      print("this is the shape of tt_weight_train", tt_weight_train.shape)
      print "inesochoa 0: this should be a weight_train = ",di_weight_train[:]
      pass

  # Read testing weights from HDF5 file
  with h5py.File('{}/Weight_1{}.h5'.format(args.output, args.nametag), 'r') as wf:
      bb_weight_test = wf['bb_vs_bb_weights'][:]
      print("this is the shape of bb_weight_test", bb_weight_test.shape)
      di_weight_test = wf['dijet_vs_bb_weights'][:]
      #print(di_weight_test)
      print("this is the shape of di_weight_test", di_weight_test.shape)
      if args.ttbar:
          tt_weight_test = wf['ttbar_vs_bb_weights'][:]
          pass
      print("this is the shape of tt_weight_test", tt_weight_test.shape)
      print "inesochoa 0: this should be a weight_test = ",di_weight_test[:]
      pass

  if args.ttbar:
    # features
    X = np.concatenate((bbjets, dijets, ttbar))
    # classes # assuming dijet and ttbar are same (1)
    Y = np.concatenate((np.zeros(len(bbjets)),np.ones(len(dijets)+len(ttbar))))
    # weights
    W_train = np.concatenate(( bb_weight_train, di_weight_train, tt_weight_train))
    W_test  = np.concatenate(( bb_weight_test,  di_weight_test,  tt_weight_test))
  else:
    # features
    X = np.concatenate((bbjets, dijets))
    # classes # assuming dijet and ttbar are same (1)
    Y = np.concatenate((np.zeros(len(bbjets)),np.ones(len(dijets))))
    # weights
    W_train = np.concatenate(( bb_weight_train, di_weight_train))
    W_test  = np.concatenate(( bb_weight_test,  di_weight_test))
    #print W[:]
    pass

  # Shuffle arrays
  indices = np.arange(X.shape[0], dtype=int)
  np.random.shuffle(indices)

  X = X[indices]
  Y = Y[indices]
  W_train = W_train[indices]
  W_test  = W_test [indices]

  # Read variable list from file
  with open(args.output+'/variables.txt','r') as varfile:
    var_list = varfile.read().splitlines()
    pass

  # Multiply re-weighing weights with MC weights
  W_train *= X[:,var_list.index('weight')]
  W_test  *= X[:,var_list.index('weight')]

  # ...
  n_mv2c = len(variable_info.default_vars)*2 #FIXME: missing _trk_? -> FIXME: make it depend on number of subjets
  n_fatjet = len(variable_info.fat_jet_vars)
  n_label = 2  # label, DSID

  ini_feature = n_mv2c + n_fatjet + n_label

  frac = [0.8,0.9]
  n_total   = len(var_list)
  n_feature = len(var_list[ini_feature:])

  # label
  arr_label = X[:,var_list.index("label")]
  #arr_label_ttbar=ttbar[:,0]

  # fat-jet info + baseline tagger
  arr_jet_pt = X[:,var_list.index("fat_jet_pt")]
  arr_jet_m = X[:,var_list.index("fat_jet_mass")]
  tmp = var_list.index(args.subjet+"_1_"+"MV2c10_discriminant")
  arr_baseline_tagger = X[:,tmp:tmp+n_mv2c] #don't hard code this!
  # features
  X = X[:,ini_feature:]
  print "inesochoa 0: this should pT = ",X[:,0]

  # train, test and validation sets
  # features
  zeros = np.zeros((X.shape[0],), dtype=bool)
  train = np.array(zeros) 
  test  = np.array(zeros) 
  val   = np.array(zeros)
  
  train[:int(frac[0]*X.shape[0])] = True
  test [int(frac[0]*X.shape[0])+1:int(frac[1]*X.shape[0])] = True
  val  [int(frac[1]*X.shape[0])+1:] = True

  # --- finally, prepare output h5
  mean = np.mean(X[train], axis=0)
  std  = np.std (X[train], axis=0)

  # Save to HDF5
  # --------------------------
  datasets = {
    # Training arrays
    'X':       X,
    'Y':       Y,
    'W_train': W_train,
    'W_test':  W_test,

    # Masks
    'train': train,
    'test':  test,
    'val':   val,

    # Per-clas data
    'dijet': dijets,
    'bbjet': bbjets,
    'dijet_weight': di_weight,
    'bbjet_weight': bb_weight,

    # Feature scaling
    'mean': mean,
    'std':  std,

    # Reference features
    'arr_baseline_tagger': arr_baseline_tagger,
    'arr_jet_pt': arr_jet_pt,
    'arr_jet_m':  arr_jet_m,
    'arr_label':  arr_label,
    }
  if args.ttbar:
    datasets['ttbar']        = ttbar
    datasets['ttjet_weight'] = tt_weight
    pass
  
  # Actually save stuff
  save_to_hdf5(args.output + '/prepared_sample_no_scaling_v2.h5', datasets)

  # Do rescaling of `X`
  for i in range(n_feature-1):
    if (std[i]!=0 and bool(args.scaling)):
      X[:,i]=(X[:,i]-mean[i])/std[i]
      pass
    pass

  # Update `X` field
  datasets['X'] = X

  # Save to HDF5
  save_to_hdf5(args.output+'/prepared_sample_v2.h5', datasets)
  pass


# Main function call.
if __name__ == "__main__" :
    main()
    pass
