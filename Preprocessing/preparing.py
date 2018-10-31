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

  # Read re-weighting weights from HDF5 file
  with h5py.File('{}/Weight_0{}.h5'.format(args.output, args.nametag), 'r') as wf:
      bb_weight = wf['bb_vs_bb_weights'][:]
      print("this is the shape of bb_weight", bb_weight.shape)
      di_weight = wf['dijet_vs_bb_weights'][:]
      #print(di_weight)
      print("this is the shape of di_weight", di_weight.shape)
      if args.ttbar:
          tt_weight = wf['ttbar_vs_bb_weights'][:]
          pass
      print("this is the shape of tt_weight", tt_weight.shape)
      print "inesochoa 0: this should be a weight = ",di_weight[:]
      pass

  if args.ttbar:
    # features
    X = np.concatenate((bbjets, dijets, ttbar))
    # classes # assuming dijet and ttbar are same (1)
    Y = np.concatenate((np.zeros(len(bbjets)),np.ones(len(dijets)+len(ttbar))))
    # weights
    X_weights = np.concatenate(( bb_weight, di_weight, tt_weight))
  else:
    # features
    X = np.concatenate((bbjets, dijets))
    # classes # assuming dijet and ttbar are same (1)
    Y = np.concatenate((np.zeros(len(bbjets)),np.ones(len(dijets))))
    # weights
    X_weights = np.concatenate(( bb_weight, di_weight))
    #print X_weights[:]
    pas

  # Shuffle arrays
  indices = np.arange(X.shape[0], dtype=int)
  np.random.shuffle(indices)

  X         = X[indices]
  Y         = Y[indices]
  X_weights = X_weights[indices]

  # OK, need to do some manipulation here... looking at variables.txt for this
  varfile = open(args.output+'/variables.txt','r')
  var_list = varfile.read().splitlines()
  varfile.close()

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
  X_train = X[:int(frac[0]*X.shape[0]), ]
  X_test = X[int(frac[0]*X.shape[0])+1: int(frac[1]*X.shape[0]), ]
  X_val = X[int(frac[1]*X.shape[0])+1:, ]
  print "inesochoa 1: this should pT = ",X_train[:,0]
  print "inesochoa 1: this should pT = ",X_test[:,0]
  print "inesochoa 1: this should pT = ",X_val[:,0]

  # weights
  X_weights_train = X_weights[:int(frac[0]*X_weights.shape[0])]
  X_weights_test = X_weights[int(frac[0]*X_weights.shape[0])+1:int(frac[1]*X_weights.shape[0])]
  X_weights_val = X_weights[int(frac[1]*X_weights.shape[0])+1:]

  # classes
  Y_train = Y[:int(frac[0]*Y.shape[0])]
  Y_test = Y[int(frac[0]*Y.shape[0])+1:int(frac[1]*Y.shape[0])]
  Y_val = Y[int(frac[1]*Y.shape[0])+1:]
  print "inesochoa 1: this should be a weight = ",X_weights_train[:]
  print "inesochoa 1: and its prediction = ",Y_train[:]
  print "inesochoa 1: this should be a weight = ",X_weights_test[:]
  print "inesochoa 1: and its prediction = ",Y_test[:]
  print "inesochoa 1: this should be a weight = ",X_weights_val[:]
  print "inesochoa 1: and its prediction = ",Y_val[:]

  # fat-jet pt
  arr_jet_pt_train = arr_jet_pt[:int(frac[0]*arr_jet_pt.shape[0])]
  arr_jet_pt_test = arr_jet_pt[int(frac[0]*arr_jet_pt.shape[0])+1:int(frac[1]*arr_jet_pt.shape[0])]
  arr_jet_pt_val = arr_jet_pt[int(frac[1]*arr_jet_pt.shape[0])+1:]

  # fat-jet mass
  arr_jet_m_train = arr_jet_m[:int(frac[0]*arr_jet_m.shape[0])]
  arr_jet_m_test = arr_jet_m[int(frac[0]*arr_jet_m.shape[0])+1:int(frac[1]*arr_jet_m.shape[0])]
  arr_jet_m_val = arr_jet_m[int(frac[1]*arr_jet_m.shape[0])+1:]

  # labels
  arr_label_train = arr_label[:int(frac[0]*arr_label.shape[0])]
  arr_label_test = arr_label[int(frac[0]*arr_label.shape[0])+1:int(frac[1]*arr_label.shape[0])]
  arr_label_val = arr_label[int(frac[1]*arr_label.shape[0])+1:]

  # baseline tagger
  arr_baseline_tagger_train = arr_baseline_tagger[:int(frac[0]*arr_baseline_tagger.shape[0]), ]
  arr_baseline_tagger_test = arr_baseline_tagger[int(frac[0]*arr_baseline_tagger.shape[0])+1: int(frac[1]*arr_baseline_tagger.shape[0]), ]
  arr_baseline_tagger_val = arr_baseline_tagger[int(frac[1]*arr_baseline_tagger.shape[0])+1:, ]


  # --- finally, prepare output h5
  mean=np.mean(X_train, axis=0)
  std=np.std(X_train, axis=0)
  h5f = h5py.File(args.output+'/prepared_sample_no_scaling_v2.h5', 'w')
  # features
  h5f.create_dataset('X_train', data=X_train)
  h5f.create_dataset('X_test', data=X_test)
  h5f.create_dataset('X_val', data=X_val)

  print "inesochoa 2: this should pT = ",X_train[:,0]
  print "inesochoa 2: this should pT = ",X_test[:,0]
  print "inesochoa 2: this should pT = ",X_val[:,0]
  # labels
  h5f.create_dataset('Y_train', data=Y_train)
  h5f.create_dataset('Y_test', data=Y_test)
  h5f.create_dataset('Y_val', data=Y_val)
  # weights
  h5f.create_dataset('X_weights_train', data=X_weights_train)
  h5f.create_dataset('X_weights_test', data=X_weights_test)
  h5f.create_dataset('X_weights_val', data=X_weights_val)
  # original samples
  h5f.create_dataset('dijet', data=dijets)
  h5f.create_dataset('bbjet', data=bbjets)
  if args.ttbar: h5f.create_dataset('ttbar', data=ttbar)
  # and weights
  h5f.create_dataset('dijet_weight', data=di_weight)
  h5f.create_dataset('bbjet_weight', data=bb_weight)
  if args.ttbar: h5f.create_dataset('ttjet_weight', data=tt_weight)
  h5f.create_dataset('mean', data=mean)
  h5f.create_dataset('std', data=std)
  # jet / event info
  h5f.create_dataset('arr_baseline_tagger_train', data=arr_baseline_tagger_train)
  h5f.create_dataset('arr_baseline_tagger_val', data=arr_baseline_tagger_val)
  h5f.create_dataset('arr_baseline_tagger_test', data=arr_baseline_tagger_test)
  h5f.create_dataset('arr_jet_pt_train', data=arr_jet_pt_train)
  h5f.create_dataset('arr_jet_pt_val', data=arr_jet_pt_val)
  h5f.create_dataset('arr_jet_pt_test', data=arr_jet_pt_test)
  h5f.create_dataset('arr_jet_m_train', data=arr_jet_m_train)
  h5f.create_dataset('arr_jet_m_val', data=arr_jet_m_val)
  h5f.create_dataset('arr_jet_m_test', data=arr_jet_m_test)
  h5f.create_dataset('arr_label_train', data=arr_label_train)
  h5f.create_dataset('arr_label_val', data=arr_label_val)
  h5f.create_dataset('arr_label_test', data=arr_label_test)
  h5f.close()

  # --- scaling, prepare output h5
  for i in range(n_feature-1):
    if (std[i]!=0 and bool(args.scaling)):
      X_train[:,i]=(X_train[:,i]-mean[i])/std[i]
      X_test[:,i]=(X_test[:,i]-mean[i])/std[i]
      X_val[:,i]=(X_val[:,i]-mean[i])/std[i]

  # --- scaled output as well:
  h5f = h5py.File(args.output+'/prepared_sample_v2.h5', 'w')
  # features
  h5f.create_dataset('X_train', data=X_train)
  h5f.create_dataset('X_test', data=X_test)
  h5f.create_dataset('X_val', data=X_val)
  print "inesochoa 3: this should scaled pT = ",X_train[:,0]
  print "inesochoa 3: this should scaled pT = ",X_test[:,0]
  print "inesochoa 3: this should scaled pT = ",X_val[:,0]
  # labels
  h5f.create_dataset('Y_train', data=Y_train)
  h5f.create_dataset('Y_test', data=Y_test)
  h5f.create_dataset('Y_val', data=Y_val)
  # weights
  h5f.create_dataset('X_weights_train', data=X_weights_train)
  h5f.create_dataset('X_weights_test', data=X_weights_test)
  h5f.create_dataset('X_weights_val', data=X_weights_val)
  # original samples
  h5f.create_dataset('dijet', data=dijets)
  h5f.create_dataset('bbjet', data=bbjets)
  if args.ttbar: h5f.create_dataset('ttbar', data=ttbar)
  # and weights
  h5f.create_dataset('dijet_weight', data=di_weight)
  h5f.create_dataset('bbjet_weight', data=bb_weight)
  if args.ttbar: h5f.create_dataset('ttjet_weight', data=tt_weight)
  h5f.create_dataset('mean', data=mean)
  h5f.create_dataset('std', data=std)
  # jet / event info
  h5f.create_dataset('arr_baseline_tagger_train', data=arr_baseline_tagger_train)
  h5f.create_dataset('arr_baseline_tagger_val', data=arr_baseline_tagger_val)
  h5f.create_dataset('arr_baseline_tagger_test', data=arr_baseline_tagger_test)
  h5f.create_dataset('arr_jet_pt_train', data=arr_jet_pt_train)
  h5f.create_dataset('arr_jet_pt_val', data=arr_jet_pt_val)
  h5f.create_dataset('arr_jet_pt_test', data=arr_jet_pt_test)
  h5f.create_dataset('arr_jet_m_train', data=arr_jet_m_train)
  h5f.create_dataset('arr_jet_m_val', data=arr_jet_m_val)
  h5f.create_dataset('arr_jet_m_test', data=arr_jet_m_test)
  h5f.create_dataset('arr_label_train', data=arr_label_train)
  h5f.create_dataset('arr_label_val', data=arr_label_val)
  h5f.create_dataset('arr_label_test', data=arr_label_test)
  h5f.close()
  print'this is the part where h5 prepared sample is created'


# Main function call.
if __name__ == "__main__" :
    main()
    pass
