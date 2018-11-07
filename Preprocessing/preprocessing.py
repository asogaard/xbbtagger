'''
This the *first* script in the data preparation pipeline for the X->bb tagger
 > preprocessing.py
   reweighting.py
   preparing.py
'''

# Import(s)
import gc
import glob
import h5py
import json
import numpy as np
import pandas as pd

import utilities.variable_info as variable_info
from utilities.labelMap import label_dict, get_double_label
from utilities.common import *

# Utility method(s)
@logging
def get_dsid (filename):
    """
    Get the dataset ID from a filename.
    """
    dirname = filename.split('/')[-2]
    if '.' in dirname:
        dsid = dirname.split('.')[2]
    else:
        dsid = dirname
        pass
    return dsid


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


@garbage_collect
@logging
def read_files (args, txt):
    """
    Load data file HDF5 filesself and convert and combined to pandas.DataFrames.

    Arguments:
        args: Namespace containing command-line arguments.
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
    print "Loading samples from {:s}:".format(txt)
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
    for f in files:
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
        l_weight.append( xsection * mcEventWeight / nEventsProcessed )

        # Clean-up
        del dsid, xsection, mcEventWeight, nEventsProcessed
        gc.collect()
        pass

    # Concatenate DataFrames.
    fatjet  = pd.concat(l_fatjet,  ignore_index=True, sort=False); del l_fatjet;  gc.collect()
    subjet1 = pd.concat(l_subjet1, ignore_index=True, sort=False); del l_subjet1; gc.collect()
    subjet2 = pd.concat(l_subjet2, ignore_index=True, sort=False); del l_subjet2; gc.collect()

    # Add dataset ID column to `fatjet`
    fatjet['weight'] = np.concatenate(l_weight); del l_weight; gc.collect()

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


@garbage_collect
@logging
def decorate (df, label, scale=1):
    """
    Add additional columns to pandas.DataFrame `df`.

    Arguments:
        df: ...
        labels: ...
        scale: ...

    Returns:
        Modified pandas.DataFrame
    """

    # Add dummy `label` column and perform lookup in `label_dict` to get unique
    # integer label
    # @NOTE: Key in `label_dict` is:
    #  - 'top' is top
    #  - 'dijet' is any combination (XX)
    #  - 'signal' is H_XX (and we only keep H_bb)
    df['label'] = label
    df['label'] = df['label'].apply(lambda x: scale * label_dict[x])

    return df


@garbage_collect
@logging
def save (args, mini_data, mini_data_sj1, mini_data_sj2):
    """

    """

    # Define variables
    dataset = list()

    # ---- list of variables ---- #
    varfile = open(args.output + '/variables.txt','w')

    # Start filling dataset list below:
    # ------- label --------- #
    dataset.append( mini_data["label"] )
    varfile.write("label\n")

    # ------- event weight --------- #
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

    # Flip dataset
    flipped_dataset = np.rot90(np.array(dataset))

    # Split-up dataset into categories
    print ('Total Var = ' + str(len(dataset)))
    label  = flipped_dataset[:,0]
    bbjets = flipped_dataset[label == label_dict["H_bb"]]
    dijets = flipped_dataset[(label != label_dict["H_bb"]) & (label != label_dict["top"])]
    if args.ttbar:
        ttbar = flipped_dataset[(label == label_dict["top"])]
        pass

    # Logging
    print ( "Stats: ")
    print ( "# signal = ", len(bbjets))
    print ( "# dijets = ", len(dijets))
    if args.ttbar:
        print ( "# ttbar = ", len(ttbar))
        pass

    # Save pre-processed datasets to HDF5 file
    outfile = '{}/output_Preprocessed{}.h5'.format(args.output, args.nametag).replace('//', '/')
    print "Saving output to file\n  {}".format(outfile)
    with h5py.File(outfile, 'w') as h5f:
        h5f.create_dataset('arr_processed_bbjets', data=bbjets)
        h5f.create_dataset('arr_processed_dijets', data=dijets)
        if args.ttbar:
            h5f.create_dataset('arr_processed_ttbar', data=ttbar)
            pass
        pass

    return outfile


# Main function definition
@logging
def main ():

    # Parse command-line arguments
    args = parse_args()
    print "Command-line arguments:"
    print args

    # Define common variables
    sigtxt    = "files/signal.txt"
    dijettxt  = "files/dijet.txt"
    toptxt    = "files/top.txt"

    # Load the input data files
    fatjet_data_sig,     subjet_1_sig,   subjet_2_sig   = read_files(args, sigtxt)
    fatjet_data_dijet,   subjet_1_dijet, subjet_2_dijet = read_files(args, dijettxt)
    if args.ttbar:
        fatjet_data_top, subjet_1_top,   subjet_2_top   = read_files(args, toptxt)
        pass

    # Decorating dataframes
    labels_signal = fatjet_data_sig.apply(lambda x: get_double_label(subjet_1_sig['GhostBHadronsFinalCount'][x.name],
                                                                     subjet_1_sig['GhostCHadronsFinalCount'][x.name],
                                                                     subjet_2_sig['GhostBHadronsFinalCount'][x.name],
                                                                     subjet_2_sig['GhostCHadronsFinalCount'][x.name]), axis=1)
    fatjet_data_sig     = decorate(fatjet_data_sig,   labels_signal, scale=1E+03)
    fatjet_data_dijet   = decorate(fatjet_data_dijet, "dijet")
    if args.ttbar:
        fatjet_data_top = decorate(fatjet_data_top,   "top")
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
    if args.ttbar:
        print ("Combining with top dataset (%d):"%len(fatjet_data_top))
        print ('Total Events = %d'%(len(mini_data)+len(fatjet_data_dijet)+len(fatjet_data_top)))
        pass

    # Append background events
    mini_data     = pd.concat([mini_data,     fatjet_data_dijet], sort=False)
    mini_data_sj1 = pd.concat([mini_data_sj1, subjet_1_dijet],    sort=False)
    mini_data_sj2 = pd.concat([mini_data_sj2, subjet_2_dijet],    sort=False)
    if args.ttbar:
        mini_data    = pd.concat([mini_data,      fatjet_data_top], sort=False)
        mini_data_sj1 = pd.concat([mini_data_sj1, subjet_1_top],    sort=False)
        mini_data_sj2 = pd.concat([mini_data_sj2, subjet_2_top],    sort=False)
        pass

    print "Cross-check:"
    print ('Events in fat-jet array = '+str(len(mini_data)))
    print ('Events in subjet 1 array = %d'%(len(mini_data_sj1)))
    print ('Events in subjet 2 array = %d'%(len(mini_data_sj2)))
    mini_data.fillna(-99)

    # Save data to file
    save(args, mini_data, mini_data_sj1, mini_data_sj2)

    return

# Main function call.
if __name__ == "__main__" :
    main()
    pass
