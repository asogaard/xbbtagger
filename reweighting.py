# ----------------------------------------------------
# Ines Ochoa, August 2018
# Based on Jue Chen's scripts
#
# Perform pT / eta reweighting
# Note: currently reweighting signal to background...
# ----------------------------------------------------
import h5py
import numpy as np
import pandas as pd
from labelMap import get_double_label
import argparse, sys, json

#------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=sys.__doc__)
    parser.add_argument('-sj', '--subjet', default="subjet_ExKt2", help="Subjet collection.")
    parser.add_argument('-pt_flat', default=0, type=bool, help="Flatten pt distributions.")
    parser.add_argument('-tt', default=1, type=int, help="Include ttbar background.")
    parser.add_argument('-o', '--output', help="Output folder where to store h5.")
    parser.add_argument('-m', '--masscut', type=bool, default=False, help="Apply Higgs mass cut.")
    parser.add_argument('-pt', '--ptcut', type=int, default=1, help="Apply maximum pT cut on fat-jet.")
    return parser.parse_args()

#------------------------------------------------------------
def find_nearest1D(array,value):
     #check if value is below lowest or above higest bin edge,
     highBin = array[-1]
     lowbin = array[0]

     if value >= highBin:
         return str(array[-2])+'_'+str(array[-1])
     if value < lowbin:
         return str(array[0])+'_'+str(array[1])

     idxLow = (np.abs(array[array <= value]-value)).argmin()
     idxHigh = (np.abs(array[array > value]-value)).argmin()

     return str(array[array <= value][idxLow])+'_'+str( array[array > value][idxHigh] )

#------------------------------------------------------------
def find2dBin(xval, yval, xbins, ybins):

     xbinning = find_nearest1D(xbins,xval)
     ybinning = find_nearest1D(ybins,yval)

     return xbinning+'_'+ybinning

#------------------------------------------------------------
def makeBinValueDict(array,x_edges,y_edges):

     binvaldict = {}

     for xi in range(len(x_edges)-1):
         for yi in range(len(y_edges)-1):
             binname = str(x_edges[xi])+'_'+str(x_edges[xi+1])+'_'+str(y_edges[yi])+'_'+str(y_edges[yi+1])
             binval = array[xi][yi]
             if np.isnan(binval):
                 binval = 1.0
             binvaldict[binname]=binval
     return binvaldict

#------------------------------------------------------------
def ptflat_reweight(num_array,denom_array,ptbins=[],etabins=[]):
    #the inputs should be 2 columns, of the two variables used for reweighting

    ptarray_num = num_array[:,0]
    etaarray_num = num_array[:,1]
    weight_num = num_array[:,2]

    ptarray_denom = denom_array[:,0]
    etaarray_denom = denom_array[:,1]
    weight_denom = denom_array[:,2]

    print(len(ptbins), len(etabins))
    if len(ptbins)==0:
        ptbins = np.linspace(ptarray_num.min(),ptarray_num.max(),101)
    if len(etabins)==0:
        etabins = np.linspace(etaarray_num.min(),etaarray_num.max(),10)

    print "bins", ptbins, etabins
    h_num, xedges, yedges = np.histogram2d(ptarray_num, etaarray_num, bins=(ptbins, etabins))
    print(xedges, yedges)
    h_denom, xedges, yedges = np.histogram2d(ptarray_denom, etaarray_denom, bins=(ptbins, etabins))
    print(xedges, yedges)
    a = h_num
    b = h_denom
    print(np.shape(a))

    mean=np.divide(np.sum(a,axis=0),np.count_nonzero(b,axis=0))

    pt=np.ones(len(a))
    pt.shape=(len(a),1)
    mean.shape=(len(a[0]),1)
    mean=pt*mean.T
    weight=np.divide(mean,b)
    weightHist=weight
    weightDict = makeBinValueDict(weightHist,xedges, yedges)


    weightarray = []

    for row in denom_array:
        xval = row[0]
        yval = row[1]

        weightarray.append( weightDict[find2dBin(xval,yval,xedges,yedges)] )

    return weightarray

#------------------------------------------------------------
def ptEta_reweight(num_array,denom_array,ptbins=[],etabins=[]):
    #the inputs should be 2 columns, of the two variables used for reweighting

    ptarray_num = num_array[:,0].astype(np.float)
    etaarray_num = num_array[:,1].astype(np.float)
    weight_num = num_array[:,2].astype(np.float)

    ptarray_denom = denom_array[:,0].astype(np.float)
    etaarray_denom = denom_array[:,1].astype(np.float)
    weight_denom = denom_array[:,2].astype(np.float)

    if len(ptbins)==0:
        ptbins = np.linspace(ptarray_num.min(),ptarray_num.max(),51)
    if len(etabins)==0:
        etabins = np.linspace(etaarray_num.min(),etaarray_num.max(),11)

    h_denom, xedges, yedges = np.histogram2d(ptarray_denom, etaarray_denom, bins=(ptbins, etabins))#, weights=weight_denom)
    h_num, xedges, yedges = np.histogram2d(ptarray_num, etaarray_num, bins=(ptbins, etabins))#, weights=weight_num)

   #replace 0 values in Hdenom
    for xi in range(len(h_denom)):
        for yi in range(len(h_denom[0])):
            if h_denom[xi][yi] == 0:
                h_denom[xi][yi] = 1

    weightHist = np.divide(h_num,h_denom)


    weightDict = makeBinValueDict(weightHist,xedges, yedges)
    weightarray = []

    for row in denom_array:
        xval = row[0]
        yval = row[1]

        weightarray.append( weightDict[find2dBin(xval,yval,xedges,yedges)] )

    return weightarray


def produce_weights():

    args = parse_args()
    subjet_collection = args.subjet

    # ---- get xsection info ---- #
    with open('mc_info.json') as f: xsec_data = json.load(f)

    # ---- load the input data file ----- #
    # --- signal
    print "Preparing signal samples..."
    signal_samples = open(sigtxt,"r").read().splitlines()
    file_sig = []
    dsid_list = []
    #fsig = h5py.File('/data/users/miochoa/doubleTagger/samples/user.dguest.301503.hbbTraining.e3820_e5984_s3126_r10201_r10210_p3596.p3_output.h5/user.dguest.14784696._000001.output.h5', 'r')
    #fat_jet_tree_sig = np.array(fsig["fat_jet"][:])
    #subjet_1_sig = np.array(fsig[subjet_collection+"_1"][:])
    #subjet_2_sig = np.array(fsig[subjet_collection+"_2"][:])
    #metadata_tree_sig = np.array(fsig['metadata'])
    for sample in signal_samples:
        content = open(input_dir+sample+".txt","r").read().splitlines()
        file_sig.append(content[0].rstrip("\n").replace(" ", "")) #only using one file
        dsid_list.append(sample.split(".")[2])

    print "Found %d signal samples to combine."%len(file_sig)
    # get shape from first file
    fsig = h5py.File(file_sig[0], 'r')
    print(fsig["fat_jet"])
    fat_jet_tree_sig = np.array(fsig["fat_jet"]['eta','pt','mass','GhostHBosonsCount'])
    subjet_1_sig = np.array(fsig[subjet_collection+"_1"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])
    subjet_2_sig = np.array(fsig[subjet_collection+"_2"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])
    dsid = dsid_list[0]
    xsection = float(xsec_data[dsid]["crossSection"])*float(xsec_data[dsid]["filtereff"])
    weight_sig = np.array(fsig["fat_jet"]["mcEventWeight"][:])*xsection/fsig["metadata"]["nEventsProcessed"]
    fsig.close()

    # loop over remaining ones
    #for i,f in enumerate(file_sig[1:]):
    for i in range(1,len(file_sig)):
        dsid = dsid_list[i]
        fsig = h5py.File(file_sig[i], 'r')
        fat_jet_tree_sig = np.hstack((fat_jet_tree_sig,np.array(fsig["fat_jet"]['eta','pt','mass','GhostHBosonsCount'])))
        subjet_1_sig = np.hstack((subjet_1_sig,np.array(fsig[subjet_collection+"_1"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])))
        subjet_2_sig = np.hstack((subjet_2_sig,np.array(fsig[subjet_collection+"_2"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])))
        xsection = float(xsec_data[dsid]["crossSection"])*float(xsec_data[dsid]["filtereff"])
        weight_sig = np.hstack((weight_sig,np.array(fsig["fat_jet"]["mcEventWeight"][:])*xsection/fsig["metadata"]["nEventsProcessed"]))
        fsig.close()
    print "check size (before): %d, %d, %d"%(len(fat_jet_tree_sig),len(subjet_1_sig),len(subjet_2_sig))

    # ---
    # --- Higgs boson matching for signal -> need to do it for all collections
    subjet_1_sig = np.extract(fat_jet_tree_sig['GhostHBosonsCount']>=1,subjet_1_sig)
    subjet_2_sig = np.extract(fat_jet_tree_sig['GhostHBosonsCount']>=1,subjet_2_sig)
    weight_sig = np.extract(fat_jet_tree_sig['GhostHBosonsCount']>=1,weight_sig)
    fat_jet_tree_sig = np.extract(fat_jet_tree_sig['GhostHBosonsCount']>=1,fat_jet_tree_sig)
    print "check size (after 1): %d, %d, %d"%(len(fat_jet_tree_sig),len(subjet_1_sig),len(subjet_2_sig))

    # ---
    # --- CUTS
    if bool(args.masscut) == True:
        massCut = (fat_jet_tree_sig['mass']>=75e3) & (fat_jet_tree_sig['mass']<=145e3)
        subjet_1_sig = subjet_1_sig[massCut]
        subjet_2_sig = subjet_2_sig[massCut]
        fat_jet_tree_sig = fat_jet_tree_sig[massCut]
        weight_sig =weight_sig[massCut]
        print "check size (after 2): %d, %d, %d"%(len(fat_jet_tree_sig),len(subjet_1_sig),len(subjet_2_sig))

    if bool(args.ptcut) == True:
        ptCut = (fat_jet_tree_sig['pt']<3000e3) & (subjet_1_sig['pt']<3000e3) & (subjet_2_sig['pt']<3000e3)
        subjet_1_sig = subjet_1_sig[ptCut]
        subjet_2_sig = subjet_2_sig[ptCut]
        fat_jet_tree_sig = fat_jet_tree_sig[ptCut]
        weight_sig = weight_sig[ptCut]
    print "check size (after 3): %d, %d, %d"%(len(fat_jet_tree_sig),len(subjet_1_sig),len(subjet_2_sig))
    # ---

    # --- dijet
    print "Preparing dijet samples..."
    file_dijet = open(dijettxt,"r").read().splitlines()
    # get shape from first file
    fdijet = h5py.File(input_dir+file_dijet[0], 'r')
    fat_jet_tree_dijet = np.array(fdijet["fat_jet"][:50,'eta','pt','mass','GhostHBosonsCount'])
    subjet_1_dijet = np.array(fdijet[subjet_collection+"_1"][:50,'eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])
    subjet_2_dijet = np.array(fdijet[subjet_collection+"_2"][:50,'eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])
    dsid = file_dijet[0].split(".")[2]
    xsection = float(xsec_data[dsid]["crossSection"])*float(xsec_data[dsid]["filtereff"])
    weight_dijet = np.array(fdijet["fat_jet"][:50,"mcEventWeight"])*xsection/fdijet["metadata"]["nEventsProcessed"]
    fdijet.close()

    # loop over remaining ones
    for f in file_dijet[1:]:
        dsid = f.split(".")[2]
        fdijet = h5py.File(input_dir+f, 'r')
        fat_jet_tree_dijet = np.hstack((fat_jet_tree_dijet,np.array(fdijet["fat_jet"][:50,'eta','pt','mass','GhostHBosonsCount'])))
        subjet_1_dijet = np.hstack((subjet_1_dijet,np.array(fdijet[subjet_collection+"_1"][:50,'eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])))
        subjet_2_dijet = np.hstack((subjet_2_dijet,np.array(fdijet[subjet_collection+"_2"][:50,'eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])))
        xsection = float(xsec_data[dsid]["crossSection"])*float(xsec_data[dsid]["filtereff"])
        weight_dijet = np.hstack((weight_dijet,np.array(fdijet["fat_jet"][:50,"mcEventWeight"])*xsection/fdijet["metadata"]["nEventsProcessed"]))
        fdijet.close()

    # ---
    # --- CUTS
    if bool(args.masscut) == True:
        massCut = (fat_jet_tree_dijet['mass']>=75e3) & (fat_jet_tree_dijet['mass']<=145e3)
        subjet_1_dijet = subjet_1_dijet[massCut]
        subjet_2_dijet = subjet_2_dijet[massCut]
        fat_jet_tree_dijet = fat_jet_tree_dijet[massCut]
        weight_dijet = weight_dijet[massCut]
    print "check size (dijet): %d, %d, %d"%(len(fat_jet_tree_dijet),len(subjet_1_dijet),len(subjet_2_dijet))

    if bool(args.ptcut) == True:
        ptCut = (fat_jet_tree_dijet['pt']<3000e3) & (subjet_1_dijet['pt']<3000e3) & (subjet_2_dijet['pt']<3000e3)
        subjet_1_dijet = subjet_1_dijet[ptCut]
        subjet_2_dijet = subjet_2_dijet[ptCut]
        fat_jet_tree_dijet = fat_jet_tree_dijet[ptCut]
        weight_dijet = weight_dijet[ptCut]
    print "check size (dijet): %d, %d, %d"%(len(fat_jet_tree_dijet),len(subjet_1_dijet),len(subjet_2_dijet))

    # --- top
    if bool(args.tt) == True:
        print "Preparing top samples..."
        top_samples = open(toptxt,"r").read().splitlines()
        file_top = []
        dsid_list = []
        #fsig = h5py.File('/data/users/miochoa/doubleTagger/samples/user.dguest.301503.hbbTraining.e3820_e5984_s3126_r10201_r10210_p3596.p3_output.h5/user.dguest.14784696._000001.output.h5', 'r')
        #fat_jet_tree_sig = np.array(fsig["fat_jet"][:])
        #subjet_1_sig = np.array(fsig[subjet_collection+"_1"][:])
        #subjet_2_sig = np.array(fsig[subjet_collection+"_2"][:])
        #metadata_tree_sig = np.array(fsig['metadata'])
        for sample in top_samples:
            content = open(input_dir+sample+".txt","r").read().splitlines()
            file_top.append(content[0].rstrip("\n").replace(" ", "")) #only using one file
            dsid_list.append(sample.split(".")[2])

        print "Found %d top samples to combine."%len(file_sig)
        # get shape from first file
        ftop = h5py.File(file_top[0], 'r')
        fat_jet_tree_top = np.array(ftop["fat_jet"]['eta','pt','mass','GhostHBosonsCount'])
        subjet_1_top = np.array(ftop[subjet_collection+"_1"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])
        subjet_2_top = np.array(ftop[subjet_collection+"_2"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])
        dsid = dsid_list[0]
        xsection = float(xsec_data[dsid]["crossSection"])*float(xsec_data[dsid]["filtereff"])
        weight_top = np.array(ftop["fat_jet"]["mcEventWeight"][:])*xsection/ftop["metadata"]["nEventsProcessed"]
        ftop.close()

        #ftop = h5py.File('/data/users/miochoa/doubleTagger/samples/user.dguest.301328.hbbTraining.e4061_s3126_r9364_r9315_p3596.p3_output.h5/user.dguest.14784799._000001.output.h5', 'r')
        #fat_jet_tree_top = np.array(ftop["fat_jet"][:])
        #subjet_1_top = np.array(ftop[subjet_collection+"_1"][:])
        #subjet_2_top = np.array(ftop[subjet_collection+"_2"][:])
        #metadata_tree_top = np.array(ftop['metadata'])
        # loop over remaining ones
        for i in range(1,len(file_top)):
            dsid = dsid_list[i]
            ftop = h5py.File(file_top[i], 'r')
            fat_jet_tree_top = np.hstack((fat_jet_tree_top,np.array(ftop["fat_jet"]['eta','pt','mass','GhostHBosonsCount'])))
            subjet_1_top = np.hstack((subjet_1_top,np.array(ftop[subjet_collection+"_1"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])))
            subjet_2_top = np.hstack((subjet_2_top,np.array(ftop[subjet_collection+"_2"]['eta','pt','GhostBHadronsFinalCount','GhostCHadronsFinalCount'])))
            xsection = float(xsec_data[dsid]["crossSection"])*float(xsec_data[dsid]["filtereff"])
            weight_top = np.hstack((weight_top,np.array(ftop["fat_jet"]["mcEventWeight"][:])*xsection/ftop["metadata"]["nEventsProcessed"]))
            ftop.close()

        # ---
        # --- CUTS
        if bool(args.masscut) == True:
            massCut = (fat_jet_tree_top['mass']>=75e3) & (fat_jet_tree_top['mass']<=145e3)
            subjet_1_top = subjet_1_top[massCut]
            subjet_2_top = subjet_2_top[massCut]
            fat_jet_tree_top = fat_jet_tree_top[massCut]
            weight_top = weight_top[massCut]
        print "check size (top): %d, %d, %d"%(len(fat_jet_tree_top),len(subjet_1_top),len(subjet_2_top))

        if bool(args.ptcut) == True:
            ptCut = (fat_jet_tree_top['pt']<3000e3) & (subjet_1_top['pt']<3000e3) & (subjet_2_top['pt']<3000e3)
            subjet_1_top = subjet_1_top[ptCut]
            subjet_2_top = subjet_2_top[ptCut]
            fat_jet_tree_top = fat_jet_tree_top[ptCut]
            weight_top = weight_top[ptCut]
        print "check size (top): %d, %d, %d"%(len(fat_jet_tree_top),len(subjet_1_top),len(subjet_2_top))

    # ---
    dataset_sig = []
    dataset_dijet = []
    dataset_top = []

    # --- for signal, determining labels at this stage
    indices = []
    for i,j in enumerate(fat_jet_tree_sig):
        nB_1 = subjet_1_sig['GhostBHadronsFinalCount'][i]
        nC_1 = subjet_1_sig['GhostCHadronsFinalCount'][i]
        nB_2 = subjet_2_sig['GhostBHadronsFinalCount'][i]
        nC_2 = subjet_2_sig['GhostCHadronsFinalCount'][i]
        double_label = get_double_label(nB_1,nC_1,nB_2,nC_2)
        if double_label == "bb": indices.append(i)

    print "---- intermediate check:"
    print "# signal events = %d"%len(fat_jet_tree_sig)
    print "# dijet events = %d"%len(fat_jet_tree_dijet)
    if bool(args.tt) == True: print "# top events = %d"%len(fat_jet_tree_top)

    # ------- Extracting relevant variables for reweighting ------- #
    # -------      eta, pt     --------- #
    for ivar in ['pt', 'eta']:
        dataset_sig.append( fat_jet_tree_sig[ivar] ) #1, #2
        dataset_dijet.append( fat_jet_tree_dijet[ivar] )
        if bool(args.tt) == True: dataset_top.append( fat_jet_tree_top[ivar] )

    #weight1=fat_jet_tree['mcEventWeight']*fat_jet_tree['Cross_Section']*fat_jet_tree['Filter_ef']/ fat_jet_tree['sumOfWeights']
    #weight_sig=fat_jet_tree_sig['mcEventWeight']/metadata_tree_sig['sumOfWeights']
    #weight_dijet=fat_jet_tree_dijet['mcEventWeight']*xsection/metadata_tree_dijet['nEventsProcessed']
    #if args.tt: weight_top=fat_jet_tree_top['mcEventWeight']/metadata_tree_top['sumOfWeights']

    # alt2 ->  this is commented out
    weight_sig=weight_sig[:]/np.sum(weight_sig)
    weight_dijet=weight_dijet[:]/np.sum(weight_dijet)
    if bool(args.tt) == True: weight_top=weight_top[:]/np.sum(weight_top)

    dataset_sig.append(weight_sig)
    dataset_dijet.append(weight_dijet)
    if bool(args.tt) == True: dataset_top.append(weight_top)

    flipped_dataset_sig = np.rot90(np.array(dataset_sig))
    flipped_dataset_dijet = np.rot90(np.array(dataset_dijet))
    if bool(args.tt) == True: flipped_dataset_top = np.rot90(np.array(dataset_top))

    # ------- Extracting bb only (signal) ------- #
    filter_bb = np.array(indices)
    flipped_bb = flipped_dataset_sig[filter_bb]

    print "--- stats check"
    print "# bb = %d"%len(flipped_bb)
    print "# dijets = %d"%len(flipped_dataset_dijet)
    if bool(args.tt) == True: print "# top = %d"%len(flipped_dataset_top)


    if args.pt_flat==True:
        print "-> p_{T} flat reweighting"
        ptbins = np.linspace(250e3,6000e3,50)
        bbWeights = ptflat_reweight(flipped_bb, flipped_bb,ptbins=ptbins)
        dijetWeights = ptflat_reweight(flipped_bb, flipped_dataset_dijet,ptbins=ptbins)
        if bool(args.tt) == True: ttbarWeights = ptflat_reweight(flipped_bb,flipped_dataset_top,ptbins=ptbins)
    else:
        dijetWeights = ptEta_reweight(flipped_bb,flipped_dataset_dijet)
        bbWeights = ptEta_reweight(flipped_bb,flipped_bb)
        if bool(args.tt) == True: ttbarWeights = ptEta_reweight(flipped_bb,flipped_dataset_top)

    h5f = h5py.File(args.output+'/Weight_'+str(args.pt_flat)+'%s.h5'%name_tag, 'w')
    h5f.create_dataset('bb_vs_bb_weights', data=bbWeights)
    h5f.create_dataset('dijet_vs_bb_weights', data=dijetWeights)
    if bool(args.tt) == True: h5f.create_dataset('ttbar_vs_bb_weights', data=ttbarWeights)

    h5f.close()

#------------------------------
if __name__ == "__main__" :

    args = parse_args()

    global name_tag
    name_tag = ""

    global input_dir, sigtxt, toptxt, dijettxt
    input_dir = "/eos/user/e/evillhau/new_double_b_tagger_ines/double-tagger-fun/Preprocessing/"
    sigtxt = "signal.txt"
    dijettxt = "dijet.txt"
    toptxt = "top.txt"
    #try:
        #os.mkdir(args.output)
        #print "Created new output directory."
    #except Exception as err: sys.exit("Directory already exists.")

    produce_weights()

