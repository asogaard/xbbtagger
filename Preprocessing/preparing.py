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
  print "[save_to_hdf5] file: {}".format(filename)
  with h5py.File(filename, 'w') as h5f:
    for name, data in datasets.iteritems():
      h5f.create_dataset(name, data=data)
      pass
    pass
  return

def load_reweighting (filename, args):
  print "[load_reweighting] file: {}".format(filename)
  with h5py.File(filename, 'r') as h5f:
    bb_weights = h5f['bb_vs_bb_weights']   [:]
    di_weights = h5f['dijet_vs_bb_weights'][:]
    tt_weights = h5f['ttbar_vs_bb_weights'][:] if args.ttbar else None
    pass
  return bb_weights, di_weights, tt_weights


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

  # Read training and test weights from HDF5 file
  pattern = '{}/Weight_{{ptflat:d}}{}.h5'.format(args.output, args.nametag)
  bb_weight_train, di_weight_train, tt_weight_train = load_reweighting(pattern.format(ptflat=1), args)
  bb_weight_test,  di_weight_test,  tt_weight_test  = load_reweighting(pattern.format(ptflat=0), args)

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

  # Read variable list from file
  with open(args.output+'/variables.txt','r') as varfile:
    var_list = varfile.read().splitlines()
    pass

  # Multiply re-weighing weights with MC weights
  #W_train *= X[:,var_list.index('weight')]
  #W_test  *= X[:,var_list.index('weight')]

  # Normalise weights to unit mean within each class
  groups = [
    np.arange(len(bbjets)),    
    np.arange(len(dijets)) + len(bbjets),
    ]
  if args.ttbar:
    groups.append(np.arange(len(ttbar)) + len(dijets) + len(bbjets))
    pass

  # -- Normalise all groups to have sum 1
  for group in groups:
    W_train[group] /= W_train[group].sum()
    W_test [group] /= W_test [group].sum()
    pass

  # -- Normalise signal to have mean 1
  W_train[groups[0]] /= W_train[groups[0]].mean()
  W_test [groups[0]] /= W_test [groups[0]].mean()

  # -- Normalise all background groups to have same sum as signal
  for group in groups[1:]:
    W_train[group] *= W_train[groups[0]].sum() / float(len(groups) - 1)
    W_test [group] *= W_test [groups[0]].sum() / float(len(groups) - 1)
    pass

  # Shuffle arrays
  indices = np.arange(X.shape[0], dtype=int)
  np.random.shuffle(indices)

  X = X[indices]
  Y = Y[indices]
  W_train = W_train[indices]
  W_test  = W_test [indices]

  # ...
  n_mv2c   = len(variable_info.default_vars)*2 #FIXME: missing _trk_? -> FIXME: make it depend on number of subjets
  n_fatjet = len(variable_info.fat_jet_vars)
  n_label  = 2  # label, weight

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
  arr_jet_eta = X[:,var_list.index("fat_jet_eta")]
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
    'dijet_weight_train': di_weight_train,
    'bbjet_weight_train': bb_weight_train,
    'dijet_weight_test':  di_weight_test,
    'bbjet_weight_test':  bb_weight_test,

    # Feature scaling
    'mean': mean,
    'std':  std,

    # Reference features
    'arr_baseline_tagger': arr_baseline_tagger,
    'arr_jet_pt': arr_jet_pt,
    'arr_jet_mass':  arr_jet_m,
    'arr_jet_eta':  arr_jet_eta,
    'arr_label':  arr_label,
    }
  if args.ttbar:
    datasets['ttbar']              = ttbar
    datasets['ttjet_weight_train'] = tt_weight_train
    datasets['ttjet_weight_test']  = tt_weight_test
    pass
  
  # Actually save stuff
  save_to_hdf5(args.output + '/prepared_sample_no_scaling_v2.h5', datasets)

  # Do rescaling of `X`
  for i in range(n_feature-1):
    if (std[i]!=0 and bool(args.scaling)):
      X[:,i] = (X[:,i] - mean[i]) / std[i]
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
