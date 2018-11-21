import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import h5py
import argparse, sys

#------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=sys.__doc__)
    parser.add_argument('-i', '--input', help="Folder with h5 files and where to store plots.")
    parser.add_argument('-tt', default=1, type=int, help="Include top background.")
    return parser.parse_args()

#------------------------------------------------------------    
def make_plots(file_name):
  # ---- list of variables including pT, eta ---- #
  varfile = open(args.input+'/variables.txt','r')
  variablelist = varfile.read().splitlines()
  varfile.close()

  print ('Var Number = %i' % len(variablelist))

  f = h5py.File(args.input+'/'+file_name+'.h5', 'r')
  df = h5py.File(args.input+'/Weight_0.h5', 'r')

  # get the weights ==> with reweighting or not
  di_vs_bb_weights=df['dijet_vs_bb_weights'][:] 
  bb_vs_bb_weights= df['bb_vs_bb_weights'][:]
  if bool(args.tt): tt_vs_bb_weights=df['ttbar_vs_bb_weights'][:]

  # ... and the data
  dijet=f['dijet'][:]
  bbjet=f['bbjet'][:]
  if bool(args.tt): ttbar=f['ttbar'][:]
  dijet=dijet[:, :]
  bbjet=bbjet[:, :]
  if bool(args.tt): ttbar=ttbar[:, :]

  #why do I need these?
  Traing_sample = f['X_train'][:]
  Validation_sample = f['X_val'][:]
  Testing_sample = f['X_test'][:]
  #weights = f['X_weights'][:]
  training_weights = f['X_weights_train'][:]
  testing_weights = f['X_weights_test'][:]
  val_weights = f['X_weights_val'][:]
  
  # -----------------------------------------------------
  # all variables -> including classes and non-features
  # -----------------------------------------------------
  print "----------------------------------"
  print " Signal vs Background plots"
  print "----------------------------------"
  fig, ax= plt.subplots(37, 3, figsize=(40,280))
  nbins = 50
  varcounter = -1

  for i, axobjlist in enumerate(ax):
    for j, axobj in enumerate(axobjlist):
      varcounter += 1
      if varcounter >= len(variablelist): break
      varname = variablelist[varcounter]
      # ---- get ranges
      rmax = bbjet[:,varcounter].max()
      rmin = bbjet[:,varcounter].min()
      if dijet[:,varcounter].max()>rmax: rmax = dijet[:,varcounter].max()
      if dijet[:,varcounter].min()<rmin: rmin = dijet[:,varcounter].min()
      if bool(args.tt):
        if ttbar[:,varcounter].max()>rmax: rmax = ttbar[:,varcounter].max()
        if ttbar[:,varcounter].min()<rmin: rmin = ttbar[:,varcounter].min()
      if rmin == rmax: rmax = rmax+2
      nbins = np.linspace(rmin,rmax,50)
      # ---- print info
      print varcounter, varname, ", range = %0.2f - %0.2f"%(rmin,rmax)
      if "pt" in varname or "mass" in varname:
        axobj.hist(bbjet[:,varcounter],nbins,normed=1,weights = bb_vs_bb_weights, alpha=0.3,linewidth=1.5,color='r',label='bbjets',  histtype='step' )
        axobj.hist(dijet[:,varcounter],nbins,normed=1,weights = di_vs_bb_weights, alpha=0.3,linewidth=1.5,color='b',label='dijet',  histtype='step' )
        if bool(args.tt): axobj.hist(ttbar[:,varcounter],nbins,normed=1,weights = tt_vs_bb_weights, alpha=0.3,linewidth=1.5,color='g',label='ttbar',   histtype='step' )
        axobj.set_title(varname+" [MeV]")
      else:
        #if "eta" in varname: 
          #eta_bins = [0., 0.5, 0.75, 1., 2]
          #eta_bins = [-i for i in eta_bins]+eta_bins[1:5]  
          #eta_bins=np.sort(eta_bins)
          #nbins=eta_bins
        #else:
          #nbins = np.linspace(rmin,rmax,50)
        axobj.hist(bbjet[:,varcounter],nbins, normed=1, weights = bb_vs_bb_weights, alpha=0.3,linewidth=1.5,color='r',label='bbjets',  histtype='step' )
        axobj.hist(dijet[:,varcounter],nbins, normed=1, weights = di_vs_bb_weights, alpha=0.3,linewidth=1.5,color='b',label='dijet',  histtype='step' )
        if bool(args.tt): axobj.hist(ttbar[:,varcounter],nbins, normed=1, weights = tt_vs_bb_weights, alpha=0.3,linewidth=1.5,color='g',label='ttbar',   histtype='step' )
        axobj.set_title(variablelist[varcounter])
      axobj.set_yscale('log')
      axobj.set_ylabel("A.U.")
      axobj.legend()
  plt.tight_layout()

  plt.savefig(args.input+'/all_variables_signalVSbackground_%s.pdf'%file_name)

  plt.close()

  # --------------------------------------------------------
  # features only => training, test and validation samples
  # --------------------------------------------------------
  print "----------------------------------"
  print " Training vs Testing vs Validation plots"
  print "----------------------------------"
  fig, ax = plt.subplots(34, 3, figsize=(40,280))
  nbins = 50
  n_total = len(variablelist)
  ini_feature = 6 #label, fatjet pt, fatjet mass, MV2c10 1 and 2
  varcounter = ini_feature-1
  for i, axobjlist in enumerate(ax):
    for j, axobj in enumerate(axobjlist):
      varcounter += 1
      featurecounter = varcounter-ini_feature
      if varcounter >= len(variablelist): break
      varname = variablelist[varcounter]
      # ---- get ranges
      rmax = Traing_sample[:,featurecounter].max()
      rmin = Traing_sample[:,featurecounter].min()
      if rmin == rmax: rmax = rmax+2
      nbins = np.linspace(rmin,rmax,50)
      # ---- print info
      print featurecounter, varcounter, varname, ", range = %0.2f - %0.2f"%(rmin,rmax)
      if "pt" in varname or "mass" in varname:
        axobj.hist(Traing_sample[:,featurecounter],nbins,normed=1,weights = training_weights, alpha=0.3,linewidth=1.5,color='r',label='train',  histtype='step' )
        axobj.hist(Validation_sample[:,featurecounter],nbins,normed=1,weights = val_weights, alpha=0.3,linewidth=1.5,color='b',label='val',  histtype='step' )
        axobj.hist(Testing_sample[:,featurecounter],nbins,normed=1,weights = testing_weights, alpha=0.3,linewidth=1.5,color='g',label='test',   histtype='step' )
        axobj.set_title(varname+" [MeV]")
      else:
        axobj.hist(Traing_sample[:,featurecounter],nbins,normed=1,weights = training_weights, alpha=0.3,linewidth=1.5,color='r',label='train',  histtype='step' )
        axobj.hist(Validation_sample[:,featurecounter],nbins,normed=1,weights = val_weights, alpha=0.3,linewidth=1.5,color='b',label='val',  histtype='step' )
        axobj.hist(Testing_sample[:,featurecounter],nbins,normed=1,weights = testing_weights, alpha=0.3,linewidth=1.5,color='g',label='test',   histtype='step' )
        axobj.set_title(variablelist[varcounter])
      axobj.legend()
      axobj.set_yscale('log')
      axobj.set_ylabel("A.U.")
  plt.tight_layout()

  plt.savefig(args.input+'/features_%s.pdf'%file_name)

  plt.close()

#------------------------------
if __name__ == "__main__" :

  args = parse_args()
  make_plots("prepared_sample_no_scaling_v2")
  make_plots("prepared_sample_v2")