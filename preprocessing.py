# Import(s)
import gc
import sys
import glob
import h5py
import json
import numpy as np
import pandas as pd
import pickle
import random

import utilities.variable_info as variable_info
from utilities.labelMap import label_dict, get_double_label
from utilities.common import *

'''
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
  if args.tt: tt_weight= wf['ttbar_vs_bb_weights'][:]
  print("this is the shape of tt_weight", tt_weight.shape)
  print "inesochoa 0: this should be a weight = ",di_weight[:]

  if args.tt:
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
  if args.tt: h5f.create_dataset('ttbar', data=ttbar)
  # and weights
  h5f.create_dataset('dijet_weight', data=di_weight)
  h5f.create_dataset('bbjet_weight', data=bb_weight)
  if args.tt: h5f.create_dataset('ttjet_weight', data=tt_weight)
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
  if args.tt: h5f.create_dataset('ttbar', data=ttbar)
  # and weights
  h5f.create_dataset('dijet_weight', data=di_weight)
  h5f.create_dataset('bbjet_weight', data=bb_weight)
  if args.tt: h5f.create_dataset('ttjet_weight', data=tt_weight)
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
  return jetVec'''

# @TODO: Move the above to a separate script:
# (1) Preprocess (corresponding to `extract_info`)
# (2) Reweighting
# (3) Prepare training arrays

def get_dsid (filename):
    dirname = filename.split('/')[-2]
    if '.' in dirname:
        dsid = dirname.split('.')[2]
    else:
        dsid = dirname
        pass
    return dsid


def read_files (txt):
    """
    Load data file HDF5 filesself and convert and combined to pandas.DataFrames.

    Arguments:
        txt: File containing a list of directories from which to read all
            available HDF5 files, assumed to be located under `args.input`.

    Returns:
        Tuple of three pandas.Dataframes, with data for (1) the fatjet, (2) the
        leading subjet, and (3) the subleading subjet.
    """
    # Definitions
    subjet_collection = args.subjet

    variables_fatjet = variable_info.fat_jet_vars + ['GhostHBosonsCount']
    variables_subjet = variable_info.default_vars + variable_info.prob_vars + \
                       variable_info.jetfitter_vars + variable_info.SV1_vars + \
                       variable_info.kin_vars + ['GhostBHadronsFinalCount', 'GhostCHadronsFinalCount']

    # Get list of HDF5 files to read in.
    print "Loading samples form {:s}:".format(txt)
    files = []
    samples = open(txt, 'r').read().splitlines()
    for sample in samples:
        files = files + sorted(glob.glob('{}/{}/*.h5'.format(args.input, sample)))
        pass

    # Load cross-section information from file
    with open('files/mc_info.json') as f:
        xsec_data = json.load(f)
        pass

    # Read in HDF5 files as DataFrames.
    l_fatjet  = []
    l_subjet1 = []
    l_subjet2 = []
    l_weight  = []

    print "  Looping over a total of {:d} files.".format(len(files))
    for f in files[:2]:
        l_fatjet .append( pd.read_hdf(f, "fat_jet",                columns=variables_fatjet) )
        l_subjet1.append( pd.read_hdf(f, subjet_collection + "_1", columns=variables_subjet) )
        l_subjet2.append( pd.read_hdf(f, subjet_collection + "_2", columns=variables_subjet) )

        # Compute event weight
        dsid = get_dsid(f)
        xsection = float(xsec_data[dsid]["crossSection"]) * float(xsec_data[dsid]["filtereff"])
        with h5py.File(f, 'r') as h5f:
            mcEventWeight    = h5f['fat_jet']['mcEventWeight'][:]
            nEventsProcessed = h5f['metadata']['nEventsProcessed']
            pass

        weight = xsection * mcEventWeight / nEventsProcessed
        l_weight.append(weight)
        pass

    # Concatenate DataFrames.
    fatjet  = pd.concat(l_fatjet,  ignore_index=True, sort=False)
    subjet1 = pd.concat(l_subjet1, ignore_index=True, sort=False)
    subjet2 = pd.concat(l_subjet2, ignore_index=True, sort=False)

    # Add dataset ID column to `fatjet`
    fatjet['weight'] = np.concatenate(l_weight)

    # (Opt.) Apply mass cut
    print "  Total number of events before cuts: {:d}".format(len(fatjet))
    if args.masscut:
        massCut = (fatjet['mass'] >= 75.0E+03) & (fatjet['mass'] <= 145.0E+03)
        fatjet  = fatjet [massCut]
        subjet1 = subjet1[massCut]
        subjet2 = subjet2[massCut]
        print "  Total number of events after mass cut: {:d}".format(len(fatjet))
    else:
        print "  Not applying mass cut."
        pass

    # (Opt.) Apply pT cut
    if args.ptcut:
        ptCut = (fatjet['pt'] < 3000.0E+03) & (subjet1['pt'] < 3000.0E+03) & (subjet2['pt'] < 3000.0E+03)
        fatjet  = fatjet [ptCut]
        subjet1 = subjet1[ptCut]
        subjet2 = subjet2[ptCut]
        print "  Total number of events after pt cut: {:d}".format(len(fatjet))
    else:
        print "  Not applying pT cut."
        pass

    print ""

    return fatjet, subjet1, subjet2


# Load data, apply cuts, combine datasets.
def extract_info ():

    # Define common variables
    sigtxt    = "files/signal.txt"
    dijettxt  = "files/dijet.txt"
    toptxt    = "files/top.txt"

    dataset = [] # this is the dataset that will be split into bb and dijet?

    # Load the input data files
    fatjet_data_sig,   subjet_1_sig,   subjet_2_sig   = read_files(sigtxt)
    fatjet_data_dijet, subjet_1_dijet, subjet_2_dijet = read_files(dijettxt)
    if args.tt:
        fatjet_data_top,   subjet_1_top,   subjet_2_top   = read_files(toptxt)
        pass

    # Adding flavour labels
    #
    # - top is top
    # - dijet is any combination (XX)
    # - signal is H_XX (and we only keep H_bb)
    print "Adding flavour labels"

    # -- Signal
    fatjet_data_sig['str_label'] = fatjet_data_sig.apply(lambda x: get_double_label(subjet_1_sig['GhostBHadronsFinalCount'][x.name],
                                                                                    subjet_1_sig['GhostCHadronsFinalCount'][x.name],
                                                                                    subjet_2_sig['GhostBHadronsFinalCount'][x.name],
                                                                                    subjet_2_sig['GhostCHadronsFinalCount'][x.name]), axis=1)
    fatjet_data_sig['label'] = fatjet_data_sig['str_label'].apply(lambda x: 1e3*label_dict[x])

    # -- Dijet
    fatjet_data_dijet['str_label'] = "dijet"
    fatjet_data_dijet['label'] = fatjet_data_dijet['str_label'].apply(lambda x: label_dict[x])

    # -- ttbar
    if args.tt:
        fatjet_data_top['str_label'] = "top"
        fatjet_data_top['label'] = fatjet_data_top['str_label'].apply(lambda x: label_dict[x])
        pass

    print ""

    # filter signal data to include only jets matched to a bb pair and to be
    # ghost matched to a truth Higgs bosoh
    print "Filtering signal arrays so as to include only Higgs and bb matched jets..."
    print "Original number of events: %d"%len(fatjet_data_sig)
    msk = (fatjet_data_sig['label'] == label_dict['H_bb']) & (fatjet_data_sig["GhostHBosonsCount"] >= 1)
    mini_data     = fatjet_data_sig[msk]
    mini_data_sj1 = subjet_1_sig[msk]
    mini_data_sj2 = subjet_2_sig[msk]
    print "Filtered number of signal events: %d"%len(mini_data)

    print ("Combining with dijet dataset (%d):"%len(fatjet_data_dijet))
    print ('Total Events = %d'%(len(mini_data)+len(fatjet_data_dijet)))
    if args.tt:
        print ("Combining with top dataset (%d):"%len(fatjet_data_top))
        print ('Total Events = %d'%(len(mini_data)+len(fatjet_data_dijet)+len(fatjet_data_top)))
        pass

    # append background events
    mini_data=pd.concat([mini_data, fatjet_data_dijet], sort=False)
    if args.tt: mini_data=pd.concat([mini_data, fatjet_data_top], sort=False)

    mini_data_sj1=pd.concat([mini_data_sj1, subjet_1_dijet], sort=False)
    if args.tt: mini_data_sj1=pd.concat([mini_data_sj1, subjet_1_top], sort=False)

    mini_data_sj2=pd.concat([mini_data_sj2, subjet_2_dijet], sort=False)
    if args.tt: mini_data_sj2=pd.concat([mini_data_sj2, subjet_2_top], sort=False)

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

    # ------- DSID --------- #
    dataset.append( mini_data["weight"] )
    varfile.write("weight\n")

    # ------- fat-jet info --------- #
    for ivar in variable_info.fat_jet_vars:
        if "pt" in ivar or "eta" in ivar: dataset.append( mini_data[ivar]) #no scaling to the mean for these two => but there's no mean scaling here... only for NaN!
        else: dataset.append(Mean_Scale( mini_data[ivar], ivar) )
        varfile.write("fat_jet_%s\n"%ivar)
        pass

    # ------- MV2c10 variables --------- #
    for ivar in variable_info.default_vars:
        # subjet 1
        dataset.append( mini_data_sj1[ivar] )
        varfile.write( args.subjet + "_1_%s\n"%ivar)
        # subjet 2
        dataset.append( mini_data_sj2[ivar] )
        varfile.write( args.subjet + "_2_%s\n"%ivar)
        pass

    gc.collect()

    # ------- sub-jet info --------- #
    for ivar in variable_info.kin_vars:
        # subjet 1
        #print "ivar: ",mini_data_sj1[ivar],Mean_Scale( mini_data_sj1[ivar],ivar)
        dataset.append( Mean_Scale( mini_data_sj1[ivar], ivar) )
        varfile.write( args.subjet + "_1_%s\n"%ivar)
        # subjet 2
        #print "ivar: ",mini_data_sj2[ivar],Mean_Scale( mini_data_sj2[ivar],ivar)
        dataset.append( Mean_Scale( mini_data_sj2[ivar], ivar) )
        varfile.write( args.subjet + "_2_%s\n"%ivar)
        pass

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
            varfile.write( args.subjet + "_1_%s\n"%ivar)
            varfile.write( args.subjet + "_1_JetFitter_mass_check\n")
            # subjet 2
            JetFitter_mass_check_2 = FindCheck(mini_data_sj2[ivar], ivar)
            dataset.append(JetFitter_mass_check_2)
            dataset.append(Mean_Scale( mini_data_sj2[ivar], ivar))
            varfile.write( args.subjet + "_2_%s\n"%ivar)
            varfile.write( args.subjet + "_2_JetFitter_mass_check\n")
            del JetFitter_mass_check_1, JetFitter_mass_check_2
        else:
            dataset.append( Mean_Scale(mini_data_sj1[ivar], ivar))
            varfile.write(args.subjet + "_1_"+ivar+'\n')
            dataset.append( Mean_Scale(mini_data_sj2[ivar], ivar))
            varfile.write(args.subjet + "_2_"+ivar+'\n')
            pass
        pass

    gc.collect()

    # ------- SV1 variables ------------ #
    for ivar in variable_info.SV1_vars:
        if ivar == "SV1_masssvx":
            # subjet 1
            JetFitter_mass_check_1 = FindCheck(mini_data_sj1[ivar], ivar)
            dataset.append(JetFitter_mass_check_1)
            dataset.append(Mean_Scale( mini_data_sj1[ivar], ivar))
            varfile.write( args.subjet + "_1_%s\n"%ivar)
            varfile.write( args.subjet + "_1_SV1_masssvx_check\n")
            # subjet 2
            JetFitter_mass_check_2 = FindCheck(mini_data_sj2[ivar], ivar)
            dataset.append(JetFitter_mass_check_2)
            dataset.append(Mean_Scale( mini_data_sj2[ivar], ivar))
            varfile.write( args.subjet + "_2_%s\n"%ivar)
            varfile.write( args.subjet + "_2_SV1_masssvx_check\n")
            del JetFitter_mass_check_1, JetFitter_mass_check_2
        else:
            dataset.append( Mean_Scale(mini_data_sj1[ivar], ivar))
            varfile.write(args.subjet + "_1_"+ivar+'\n')
            dataset.append( Mean_Scale(mini_data_sj2[ivar], ivar))
            varfile.write(args.subjet + "_2_"+ivar+'\n')
            pass
        pass

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
        varfile.write(args.subjet + "_1_"+ivar+'\n')
        varfile.write(args.subjet + "_1_"+ivar+"_nan_check\n")

        # subjet 2
        ivar_nan = MakeNan(mini_data_sj2[ivar],ivar)
        ivar_nan_check = FindCheck(ivar_nan, ivar)
        dataset.append( Mean_Scale(ivar_nan , ivar))
        dataset.append( ivar_nan_check )
        varfile.write(args.subjet + "_2_"+ivar+'\n')
        varfile.write(args.subjet + "_2_"+ivar+"_nan_check\n")

        del ivar_nan; del ivar_nan_check;
        pass

    gc.collect()

    # flip dataset
    flipped_dataset = np.rot90(np.array(dataset))

    print ('Total Var = ' + str(len(dataset)))

    bbjets = flipped_dataset[flipped_dataset[:,0]==label_dict["H_bb"]]
    dijets = flipped_dataset[(flipped_dataset[:,0]!=label_dict["H_bb"])&(flipped_dataset[:,0]!=label_dict["top"])]
    if args.tt: ttbar = flipped_dataset[(flipped_dataset[:,0]==label_dict["top"])]

    print ( "Stats: ")
    print ( "# signal = ", len(bbjets))
    print ( "# dijets = ", len(dijets))
    if args.tt: print ( "# ttbar = ", len(ttbar))

    h5f = h5py.File(args.output+'/output_Preprocessed%s.h5'%name_tag, 'w')
    h5f.create_dataset('arr_processed_bbjets', data=bbjets)
    h5f.create_dataset('arr_processed_dijets', data=dijets)
    if args.tt: h5f.create_dataset('arr_processed_ttbar', data=ttbar)

    h5f.close()

    varfile.close()

    return


# Define main function.
def main ():

    # Parse command-line arguments
    global args
    args = parse_args()

    global name_tag
    name_tag = ""

    # start by extracting information
    extract_info()
    #print "==[ Stopping early ]====="
    #exit()
    # extra definitions for arrays
    global frac, ini_feature, n_total

    n_mv2c = len(variable_info.default_vars)*2 #FIXME: missing _trk_? -> FIXME: make it depend on number of subjets
    n_fatjet = len(variable_info.fat_jet_vars)
    n_label = 2  # label, DSID

    ini_feature = n_mv2c + n_fatjet + n_label
    varfile = open(args.output+'/variables.txt','r')
    var_list = varfile.read().splitlines()
    varfile.close()
    n_total = len(var_list)
    n_feature = len(var_list[ini_feature:])
    print 'total number of variables (not only features) = %d'%n_total
    print 'feature index = %d + %d + %d = %d'%(n_mv2c,n_fatjet,n_label,ini_feature)
    print '-> total number of features = %d'%(n_feature)
    '''

    # then send the h5 file to prepare_samples
    h5f = h5py.File(args.output+'/output_ %s.h5'%name_tag, 'r')
    bbjets=h5f['arr_processed_bbjets'][:]
    dijets=h5f['arr_processed_dijets'][:]
    #print(len(h5f['arr_processed_dijets']))
    #exit()
    if args.tt: ttbar=h5f['arr_processed_ttbar'][:]
    else: ttbar=bbjets #send dummy
    prepare_sample(bbjets, dijets, ttbar)

    h5f.close()
    '''
    return

# Main function call.
if __name__ == "__main__" :
    main()
    pass
