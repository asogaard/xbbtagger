import h5py
import numpy as np
import gc #garbage collecton
import pandas as pd
import json, pickle
import random
from labelMap import label_dict, get_double_label
import sys,argparse
# ----  default values and list of variables ----- #
import variable_info

#------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=sys.__doc__)
    parser.add_argument('-sj', '--subjet', default="subjet_ExKt2", help="Subjet collection.")
    parser.add_argument('--scaling', default=True, type=bool, help="Perform scaling.")
    parser.add_argument('-o', '--output', help="Output folder where to store h5.")
    parser.add_argument('-m', '--masscut', type=bool, default=False, help="Apply Higgs mass cut.")
    parser.add_argument('-pt', '--ptcut', type=int, default=True, help="Apply maximum pT cut on fat-jet.")
    parser.add_argument('-tt', default=1, type=int, help="Include ttbar background.")
    return parser.parse_args()

#-------------------------------------------

def prepare_sample(bbjets, dijets, ttbar):

  print "----------------------------------"
  print " Launching prepare_sample!"
  print "----------------------------------"

  args = parse_args()

  wf = h5py.File(args.output+'/Weight_0%s.h5'%name_tag, 'r')
  bb_weight= wf['bb_vs_bb_weights'][:]
  print("this is the shape of bb_weight", bb_weight.shape)
  di_weight= wf['dijet_vs_bb_weights'][:]
  print(di_weight)
  print("this is the shape of di_weight", di_weight.shape)
  if bool(args.tt) == True: tt_weight= wf['ttbar_vs_bb_weights'][:]
  print("this is the shape of tt_weight", tt_weight.shape)
  print "inesochoa 0: this should be a weight = ",di_weight[:]

  if bool(args.tt) == True:
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
    print X_weights[:]


  # put everything together and shuffle
  Y.shape = (Y.shape[0],1)
  X_weights.shape = (X_weights.shape[0],1)
  print "before reshufling"
  print X[:]
  print(X.shape)
  print Y[:]
  print(Y.shape)
  print X_weights[:]
  print(X_weights.shape)
  Z = np.hstack((X, Y, X_weights))
  print "this is Z:"
  print Z

  seed=random.randint(0,1e6)
  random.seed(seed)
  np.random.shuffle(Z)

  # OK, need to do some manipulation here... looking at variables.txt for this
  varfile = open(args.output+'/variables.txt','r')
  var_list = varfile.read().splitlines()
  varfile.close()

  #n_total = len(var_list)# - ini_feature + 1
  n_feature = len(var_list[ini_feature:])

  # this is hard-coded, but should *not* change: n_total covers all features and actual variables, while the other two come from Y and X_weights (see above definition for Z)
  X=Z[:,:n_total]
  Y=Z[:,n_total]
  X_weights=Z[:,n_total+1]

  print "after reshufling"
  print X[:]
  print Y[:]
  print X_weights[:]
  for el in X_weights[:]:
    print "%0.3f"%el

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
  if bool(args.tt) == True: h5f.create_dataset('ttbar', data=ttbar)
  # and weights
  h5f.create_dataset('dijet_weight', data=di_weight)
  h5f.create_dataset('bbjet_weight', data=bb_weight)
  if bool(args.tt) == True: h5f.create_dataset('ttjet_weight', data=tt_weight)
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
  if bool(args.tt) == True: h5f.create_dataset('ttbar', data=ttbar)
  # and weights
  h5f.create_dataset('dijet_weight', data=di_weight)
  h5f.create_dataset('bbjet_weight', data=bb_weight)
  if bool(args.tt) == True: h5f.create_dataset('ttjet_weight', data=tt_weight)
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


def MakeNan(jetVec,var) :
  jetVec_new=np.where(np.isnan(jetVec),variable_info.default_values[var], jetVec )
  return jetVec_new

def FindCheck(jetVec, var) :
  default_location = np.where(jetVec == variable_info.default_values[var])
  jet_feature_check = np.zeros(len(jetVec))
  jet_feature_check[default_location] = 1
  return jet_feature_check

def Mean_Scale(jetVec, var) :
  jetVec=np.array(jetVec)
  # ----- removing the defaults to mean ----- #
  jetVec[np.where(jetVec==variable_info.default_values[var])] = np.mean(jetVec[np.where(jetVec!=variable_info.default_values[var])])
  return jetVec

def extract_info():#sigfile,dijet_path,ttbarfile):

  #args = parse_args()
  subjet_collection = args.subjet

  dataset = [] # this is the dataset that will be split into bb and dijet?

  # ---- load the input data files ----- #
  #--- signal
  print "Starting with signal samples..."
  file_sig = []
  signal_samples = open(sigtxt,"r").read().splitlines()
  for sample in signal_samples:
    content = open(input_dir+sample+".txt","r").read().splitlines()
    file_sig.append(content[0].rstrip("\n").replace(" ", "")) #only using one file

  l_fatjet = []
  l_subjet1 = []
  l_subjet2 = []
  print "...looping over a total of %d files..."%len(file_sig)
  for f in file_sig:
    fatjet_data_sig_tmp = pd.read_hdf(f,"fat_jet")
    subjet_1_sig_tmp = pd.read_hdf(f,subjet_collection+"_1")
    subjet_2_sig_tmp = pd.read_hdf(f,subjet_collection+"_2")
    l_fatjet.append(fatjet_data_sig_tmp)
    l_subjet1.append(subjet_1_sig_tmp)
    l_subjet2.append(subjet_2_sig_tmp)

  fatjet_data_sig = pd.concat(l_fatjet,ignore_index=True)
  subjet_1_sig = pd.concat(l_subjet1,ignore_index=True)
  subjet_2_sig = pd.concat(l_subjet2,ignore_index=True)

  print "Total number of signal events before mass cut: %d"%len(fatjet_data_sig)
  if bool(args.masscut) == True:
    massCut = (fatjet_data_sig['mass']>=75e3) & (fatjet_data_sig['mass']<=145e3)
    subjet_1_sig = subjet_1_sig[massCut]
    subjet_2_sig = subjet_2_sig[massCut]
    fatjet_data_sig = fatjet_data_sig[massCut]
  print "Total number of signal events after mass cut: %d"%len(fatjet_data_sig)
  if bool(args.ptcut) == True:
    ptCut = (fatjet_data_sig['pt']<3000e3) & (subjet_1_sig['pt']<3000e3) & (subjet_2_sig['pt']<3000e3)
    subjet_1_sig = subjet_1_sig[ptCut]
    subjet_2_sig = subjet_2_sig[ptCut]
    fatjet_data_sig = fatjet_data_sig[ptCut]
  print "Total number of signal events after pt cut: %d"%len(fatjet_data_sig)
  print "Done.\n"

  #--- dijet
  print "Moving to dijet samples..."
  file_dijet = open(dijettxt,"r").read().splitlines()
  l_fatjet = []
  l_subjet1 = []
  l_subjet2 = []

  print "...looping over a total of %d files..."%len(file_dijet)
  for f in file_dijet:
    fatjet_data_dijet_tmp = pd.read_hdf(input_dir+f,"fat_jet",start=0).loc[:50,:]
    subjet_1_dijet_tmp = pd.read_hdf(input_dir+f,subjet_collection+"_1",start=0).loc[:50,:]
    subjet_2_dijet_tmp = pd.read_hdf(input_dir+f,subjet_collection+"_2",start=0).loc[:50,:]
    l_fatjet.append(fatjet_data_dijet_tmp)
    l_subjet1.append(subjet_1_dijet_tmp)
    l_subjet2.append(subjet_2_dijet_tmp)

  fatjet_data_dijet = pd.concat(l_fatjet,ignore_index=True)#,verify_integrity=True)
  subjet_1_dijet = pd.concat(l_subjet1,ignore_index=True)#,verify_integrity=True)
  subjet_2_dijet = pd.concat(l_subjet2,ignore_index=True)#,verify_integrity=True)

  print "Total number of dijet events before mass cut: %d"%len(fatjet_data_dijet)
  if bool(args.masscut) == True:
    massCut = (fatjet_data_dijet['mass']>=75e3) & (fatjet_data_dijet['mass']<=145e3)
    subjet_1_dijet = subjet_1_dijet[massCut]
    subjet_2_dijet = subjet_2_dijet[massCut]
    fatjet_data_dijet = fatjet_data_dijet[massCut]
  print "Total number of dijet events after mass cut: %d"%len(fatjet_data_dijet)
  if bool(args.ptcut) == True:
    ptCut = (fatjet_data_dijet['pt']<3000e3) & (subjet_1_dijet['pt']<3000e3) & (subjet_2_dijet['pt']<3000e3)
    subjet_1_dijet = subjet_1_dijet[ptCut]
    subjet_2_dijet = subjet_2_dijet[ptCut]
    fatjet_data_dijet = fatjet_data_dijet[ptCut]
  print "Total number of dijet events after pt cut: %d"%len(fatjet_data_dijet)
  print "Done.\n"

  #--- top
  if bool(args.tt) == True:
    print "Finally, to top samples..."
    file_top = []
    topnal_samples = open(toptxt,"r").read().splitlines()
    for sample in topnal_samples:
      content = open(input_dir+sample+".txt","r").read().splitlines()
      file_top.append(content[0].rstrip("\n").replace(" ", "")) #only using one file

    l_fatjet = []
    l_subjet1 = []
    l_subjet2 = []
    print "...looping over a total of %d files..."%len(file_top)
    for f in file_top:
      fatjet_data_top_tmp = pd.read_hdf(f,"fat_jet")
      subjet_1_top_tmp = pd.read_hdf(f,subjet_collection+"_1")
      subjet_2_top_tmp = pd.read_hdf(f,subjet_collection+"_2")
      l_fatjet.append(fatjet_data_top_tmp)
      l_subjet1.append(subjet_1_top_tmp)
      l_subjet2.append(subjet_2_top_tmp)

    fatjet_data_top = pd.concat(l_fatjet,ignore_index=True)
    subjet_1_top = pd.concat(l_subjet1,ignore_index=True)
    subjet_2_top = pd.concat(l_subjet2,ignore_index=True)

    print "Total number of top events before mass cut: %d"%len(fatjet_data_top)
    if bool(args.masscut) == True:
      massCut = (fatjet_data_top['mass']>=75e3) & (fatjet_data_top['mass']<=145e3)
      subjet_1_top = subjet_1_top[massCut]
      subjet_2_top = subjet_2_top[massCut]
      fatjet_data_top = fatjet_data_top[massCut]
    print "Total number of top events after mass cut: %d"%len(fatjet_data_top)
    if bool(args.ptcut) == True:
      ptCut = (fatjet_data_top['pt']<3000e3) & (subjet_1_top['pt']<3000e3) & (subjet_2_top['pt']<3000e3)
      subjet_1_top = subjet_1_top[ptCut]
      subjet_2_top = subjet_2_top[ptCut]
      fatjet_data_top = fatjet_data_top[ptCut]
    print "Total number of top events after pt cut: %d"%len(fatjet_data_top)
    print "Done.\n"

  # --- flavor labels: str_labels and int labels
  # - top is top
  # - dijet is any combination (XX)
  # - signal is H_XX (and we only keep H_bb)
  print "Adding flavor labels..."
  print "Signal"
  fatjet_data_sig['str_label'] = fatjet_data_sig.apply(lambda x: get_double_label(subjet_1_sig['GhostBHadronsFinalCount'][x.name],
                                                                          subjet_1_sig['GhostCHadronsFinalCount'][x.name],
                                                                          subjet_2_sig['GhostBHadronsFinalCount'][x.name],
                                                                          subjet_2_sig['GhostCHadronsFinalCount'][x.name]),axis=1)
  fatjet_data_sig['label'] = fatjet_data_sig['str_label'].apply(lambda x: 1e3*label_dict[x])

  print "Dijet"
  #fatjet_data_dijet['str_label'] = fatjet_data_dijet.apply(lambda x: get_double_label(subjet_1_dijet['GhostBHadronsFinalCount'][x.name],
  #                                                                  subjet_1_dijet['GhostCHadronsFinalCount'][x.name],
  #                                                                  subjet_2_dijet['GhostBHadronsFinalCount'][x.name],
  #                                                                  subjet_2_dijet['GhostCHadronsFinalCount'][x.name]),axis=1)
  fatjet_data_dijet['str_label'] = "dijet"
  fatjet_data_dijet['label'] = fatjet_data_dijet['str_label'].apply(lambda x: label_dict[x])

  if bool(args.tt) == True:
    print "Top"
    fatjet_data_top['str_label'] = "top"
    fatjet_data_top['label'] = fatjet_data_top['str_label'].apply(lambda x: label_dict[x])
    print "Done.\n"

  # filter signal data to include only jets matched to a bb pair
  print "Filtering signal arrays so as to include only Higgs and bb matched jets..."
  print "Original number of events: %d"%len(fatjet_data_sig)
  mini_data_tmp=fatjet_data_sig[fatjet_data_sig['label']==label_dict['H_bb']]
  mini_data_sj1_tmp=subjet_1_sig[fatjet_data_sig['label']==label_dict['H_bb']]
  mini_data_sj2_tmp=subjet_2_sig[fatjet_data_sig['label']==label_dict['H_bb']]

  # filter signal data to include only jets matched to a Higgs boson
  mini_data=mini_data_tmp[mini_data_tmp["GhostHBosonsCount"]>=1]
  mini_data_sj1=mini_data_sj1_tmp[mini_data_tmp["GhostHBosonsCount"]>=1]
  mini_data_sj2=mini_data_sj2_tmp[mini_data_tmp["GhostHBosonsCount"]>=1]
  print "Filtered number of events: %d"%len(mini_data)

  print ("Combining with dijet dataset (%d):"%len(fatjet_data_dijet))
  print ('Total Events = %d'%(len(mini_data)+len(fatjet_data_dijet)))
  if bool(args.tt) == True:
    print ("Combining with top dataset (%d):"%len(fatjet_data_top))
    print ('Total Events = %d'%(len(mini_data)+len(fatjet_data_dijet)+len(fatjet_data_top)))

  # append background events
  mini_data=pd.concat([mini_data, fatjet_data_dijet])
  if bool(args.tt) == True: mini_data=pd.concat([mini_data, fatjet_data_top])

  mini_data_sj1=pd.concat([mini_data_sj1, subjet_1_dijet])
  if bool(args.tt) == True: mini_data_sj1=pd.concat([mini_data_sj1, subjet_1_top])

  mini_data_sj2=pd.concat([mini_data_sj2, subjet_2_dijet])
  if bool(args.tt) == True: mini_data_sj2=pd.concat([mini_data_sj2, subjet_2_top])

  print "Cross-check:"
  print ('Events in fat-jet array = '+str(len(mini_data)))
  print ('Events in subjet 1 array = %d'%(len(mini_data_sj1)))
  print ('Events in subjet 2 array = %d'%(len(mini_data_sj2)))
  mini_data.fillna(-99)

  # ---- list of variables ---- #
  varfile = open(args.output+'/variables.txt','w')

  # Start filling dataset list below:
  # ------- label --------- #
  dataset.append( mini_data["label"] )
  varfile.write("label\n")

  # ------- fat-jet info --------- #
  for ivar in variable_info.fat_jet_vars:
    if "pt" in ivar or "eta" in ivar: dataset.append( mini_data[ivar]) #no scaling to the mean for these two => but there's no mean scaling here... only for NaN!
    else: dataset.append(Mean_Scale( mini_data[ivar], ivar) )
    varfile.write("fat_jet_%s\n"%ivar)

  # ------- MV2c10 variables --------- #
  for ivar in variable_info.default_vars:
    # subjet 1
    dataset.append( mini_data_sj1[ivar] )
    varfile.write( subjet_collection+"_1_%s\n"%ivar)
    # subjet 2
    dataset.append( mini_data_sj2[ivar] )
    varfile.write( subjet_collection+"_2_%s\n"%ivar)
  gc.collect()

  # ------- sub-jet info --------- #
  for ivar in variable_info.kin_vars:
    # subjet 1
    #print "ivar: ",mini_data_sj1[ivar],Mean_Scale( mini_data_sj1[ivar],ivar)
    dataset.append( Mean_Scale( mini_data_sj1[ivar], ivar) )
    varfile.write( subjet_collection+"_1_%s\n"%ivar)
    # subjet 2
    #print "ivar: ",mini_data_sj2[ivar],Mean_Scale( mini_data_sj2[ivar],ivar)
    dataset.append( Mean_Scale( mini_data_sj2[ivar], ivar) )
    varfile.write( subjet_collection+"_2_%s\n"%ivar)
  gc.collect()

  # ------- extra variables --------- # #FIXME
   #, 'jet_Exkt_Subjet_deltaR'  ,'jet_Exkt_Subjet_Pt_imbalance' ] :

  # ------- Jet fitter variables ------------ #
  for ivar in variable_info.jetfitter_vars: # why is mass different?
    if ivar == "JetFitter_mass":
      # subjet 1
      JetFitter_mass_check_1 = FindCheck(mini_data_sj1[ivar], ivar)
      dataset.append(JetFitter_mass_check_1)
      dataset.append(Mean_Scale( mini_data_sj1[ivar], ivar))
      varfile.write( subjet_collection+"_1_%s\n"%ivar)
      varfile.write( subjet_collection+"_1_JetFitter_mass_check\n")
      # subjet 2
      JetFitter_mass_check_2 = FindCheck(mini_data_sj2[ivar], ivar)
      dataset.append(JetFitter_mass_check_2)
      dataset.append(Mean_Scale( mini_data_sj2[ivar], ivar))
      varfile.write( subjet_collection+"_2_%s\n"%ivar)
      varfile.write( subjet_collection+"_2_JetFitter_mass_check\n")
      del JetFitter_mass_check_1, JetFitter_mass_check_2
    else:
      dataset.append( Mean_Scale(mini_data_sj1[ivar], ivar))
      varfile.write(subjet_collection+"_1_"+ivar+'\n')
      dataset.append( Mean_Scale(mini_data_sj2[ivar], ivar))
      varfile.write(subjet_collection+"_2_"+ivar+'\n')
  gc.collect()

  # ------- SV1 variables ------------ #
  for ivar in variable_info.SV1_vars:
    if ivar == "SV1_masssvx":
      # subjet 1
      JetFitter_mass_check_1 = FindCheck(mini_data_sj1[ivar], ivar)
      dataset.append(JetFitter_mass_check_1)
      dataset.append(Mean_Scale( mini_data_sj1[ivar], ivar))
      varfile.write( subjet_collection+"_1_%s\n"%ivar)
      varfile.write( subjet_collection+"_1_SV1_masssvx_check\n")
      # subjet 2
      JetFitter_mass_check_2 = FindCheck(mini_data_sj2[ivar], ivar)
      dataset.append(JetFitter_mass_check_2)
      dataset.append(Mean_Scale( mini_data_sj2[ivar], ivar))
      varfile.write( subjet_collection+"_2_%s\n"%ivar)
      varfile.write( subjet_collection+"_2_SV1_masssvx_check\n")
      del JetFitter_mass_check_1, JetFitter_mass_check_2
    else:
      dataset.append( Mean_Scale(mini_data_sj1[ivar], ivar))
      varfile.write(subjet_collection+"_1_"+ivar+'\n')
      dataset.append( Mean_Scale(mini_data_sj2[ivar], ivar))
      varfile.write(subjet_collection+"_2_"+ivar+'\n')
  gc.collect()

  # ------- missing variables ------------ #
  #missing_vars = ['IP2D_nTrks','IP3D_nTrks'] #FIXME -> added in newer version of FTAG5
  #missing_vars = ['numConstituents'] #FIXME -> is this saved anywhere at all?

  # ------- pb,pu,pb,ptau variables ------------ #
  for ivar in variable_info.prob_vars:
    # subjet 1
    ivar_nan = MakeNan(mini_data_sj1[ivar],ivar)
    ivar_nan_check = FindCheck(ivar_nan, ivar)
    dataset.append( Mean_Scale(ivar_nan , ivar))
    dataset.append( ivar_nan_check )
    varfile.write(subjet_collection+"_1_"+ivar+'\n')
    varfile.write(subjet_collection+"_1_"+ivar+"_nan_check\n")

    # subjet 2
    ivar_nan = MakeNan(mini_data_sj2[ivar],ivar)
    ivar_nan_check = FindCheck(ivar_nan, ivar)
    dataset.append( Mean_Scale(ivar_nan , ivar))
    dataset.append( ivar_nan_check )
    varfile.write(subjet_collection+"_2_"+ivar+'\n')
    varfile.write(subjet_collection+"_2_"+ivar+"_nan_check\n")

    del ivar_nan; del ivar_nan_check;
  gc.collect()

  # flip dataset
  flipped_dataset = np.rot90(np.array(dataset))

  print ('Total Var = ' + str(len(dataset)))

  bbjets = flipped_dataset[flipped_dataset[:,0]==label_dict["H_bb"]]
  dijets = flipped_dataset[(flipped_dataset[:,0]!=label_dict["H_bb"])&(flipped_dataset[:,0]!=label_dict["top"])]
  if bool(args.tt) == True: ttbar = flipped_dataset[(flipped_dataset[:,0]==label_dict["top"])]

  print ( "Stats: ")
  print ( "# signal = ", len(bbjets))
  print ( "# dijets = ", len(dijets))
  if bool(args.tt) == True: print ( "# ttbar = ", len(ttbar))

  h5f = h5py.File(args.output+'/output_Preprocessed%s.h5'%name_tag, 'w')
  h5f.create_dataset('arr_processed_bbjets', data=bbjets)
  h5f.create_dataset('arr_processed_dijets', data=dijets)
  if bool(args.tt) == True: h5f.create_dataset('arr_processed_ttbar', data=ttbar)

  h5f.close()
  varfile.close()

#------------------------------
if __name__ == "__main__" :

  args = parse_args()

  global name_tag
  name_tag = ""

  # need to combine dijet datasets first
  global input_dir, sigtxt, toptxt, dijettxt
  input_dir = "/eos/user/e/evillhau/new_double_b_tagger_ines/double-tagger-fun/Preprocessing/"
  sigtxt = "signal.txt"
  dijettxt = "dijet.txt"
  toptxt = "top.txt"

  # start by extracting information
  extract_info()

  # extra definitions for arrays
  global frac, ini_feature, n_total
  frac = [0.8,0.9]
  n_mv2c = len(variable_info.default_vars)*2 #FIXME: missing _trk_? -> FIXME: make it depend on number of subjets
  n_fatjet = len(variable_info.fat_jet_vars)
  n_label = 1

  ini_feature = n_mv2c + n_fatjet + n_label
  varfile = open(args.output+'/variables.txt','r')
  var_list = varfile.read().splitlines()
  varfile.close()
  n_total = len(var_list)
  n_feature = len(var_list[ini_feature:])
  print 'total number of variables (not only features) = %d'%n_total
  print 'feature index = %d + %d + %d = %d'%(n_mv2c,n_fatjet,n_label,ini_feature)
  print '-> total number of features = %d'%(n_feature)


  # then send the h5 file to prepare_samples
  h5f = h5py.File(args.output+'/output_Preprocessed%s.h5'%name_tag, 'r')
  bbjets=h5f['arr_processed_bbjets'][:]
  dijets=h5f['arr_processed_dijets'][:]
  #print(len(h5f['arr_processed_dijets']))
  #exit()
  if bool(args.tt) == True: ttbar=h5f['arr_processed_ttbar'][:]
  else: ttbar=bbjets #send dummy
  prepare_sample(bbjets, dijets, ttbar)

  h5f.close()


