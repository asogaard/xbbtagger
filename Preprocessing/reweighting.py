'''
This the *second* script in the data preparation pipeline for the X->bb tagger
   preprocessing.py
 > reweighting.py
   preparing.py
'''

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

from utilities.labelMap import get_double_label
from utilities.common import *

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
def ptflat_reweight(num_array, denom_array, bins_pt=50, bins_eta=10):
    #the inputs should be 2 columns, of the two variables used for reweighting

    # Separate arrays for numerator
    num_pt       = num_array  [:,0].astype(np.float)
    num_eta      = num_array  [:,1].astype(np.float)
    num_weight   = num_array  [:,2].astype(np.float)

    # Separate arrays for denominator
    denom_pt     = denom_array[:,0].astype(np.float)
    denom_eta    = denom_array[:,1].astype(np.float)
    denom_weight = denom_array[:,2].astype(np.float)

    # Make sure that bins are set
    if isinstance(bins_pt, int):
        print "Using automatic bins for pT  (N = {})".format(bins_pt)
        bins_pt  = np.linspace(num_pt.min(),  num_pt.max(),  bins_pt  + 1, endpoint=True)
        pass

    if isinstance(bins_eta, int):
        print "Using automatic bins for eta (N = {})".format(bins_eta)
        bins_eta = np.linspace(num_eta.min(), num_eta.max(), bins_eta + 1, endpoint=True)
        pass

    # Fill 2D histograms for numerator and denominator
    # @NOTE: Use weights? Not done in original script
    h_denom, xedges, yedges = np.histogram2d(denom_pt, denom_eta, bins=(bins_pt, bins_eta))#, weights=denom_weight)
    h_num,   _,      _      = np.histogram2d(num_pt,   num_eta,   bins=(bins_pt, bins_eta))#, weights=num_weight)

    a = h_num
    b = h_denom
    #print(np.shape(a))

    mean = np.divide(np.sum(h_num, axis=0), np.count_nonzero(h_denom, axis=0))

    pt=np.ones(len(h_num))
    pt.shape=(len(h_num),1)
    mean.shape=(len(h_num[0]),1)
    mean=pt*mean.T
    weight=np.divide(mean,h_denom)
    weightHist=weight
    weightDict = makeBinValueDict(weightHist,xedges, yedges)


    weightarray = []

    for row in denom_array:
        xval = row[0]
        yval = row[1]

        weightarray.append( weightDict[find2dBin(xval,yval,xedges,yedges)] )

    return weightarray

#------------------------------------------------------------
def ptEta_reweight(num_array, denom_array, bins_pt=50, bins_eta=10):
    #the inputs should be 2 columns, of the two variables used for reweighting

    # Separate arrays for numerator
    num_pt       = num_array  [:,0].astype(np.float)
    num_eta      = num_array  [:,1].astype(np.float)
    num_weight   = num_array  [:,2].astype(np.float)

    # Separate arrays for denominator
    denom_pt     = denom_array[:,0].astype(np.float)
    denom_eta    = denom_array[:,1].astype(np.float)
    denom_weight = denom_array[:,2].astype(np.float)

    # Make sure that bins are set
    if isinstance(bins_pt, int):
        print "Using automatic bins for pT  (N = {})".format(bins_pt)
        bins_pt  = np.linspace(num_pt.min(),  num_pt.max(),  bins_pt  + 1, endpoint=True)
        pass

    if isinstance(bins_eta, int):
        print "Using automatic bins for eta (N = {})".format(bins_eta)
        bins_eta = np.linspace(num_eta.min(), num_eta.max(), bins_eta + 1, endpoint=True)
        pass

    # Fill 2D histograms for numerator and denominator
    # @NOTE: Use weights? Not done in original script
    h_denom, xedges, yedges = np.histogram2d(denom_pt, denom_eta, bins=(bins_pt, bins_eta))#, weights=denom_weight)
    h_num,   _,      _      = np.histogram2d(num_pt,   num_eta,   bins=(bins_pt, bins_eta))#, weights=num_weight)

    # Remove zeros in denominator @NOTE: Necessary/proper?
    #h_denom = np.clip(h_denom, 1E-05, None)  # @NOTE: (..., 1, ...)?

    # Take ratio
    weightHist = np.divide(h_num, h_denom)

    # Compute per-jet weights
    ixs = np.clip(np.digitize(denom_pt,  xedges) - 1, 0, len(xedges) - 2)
    iys = np.clip(np.digitize(denom_eta, yedges) - 1, 0, len(yedges) - 2)
    reweights = weightHist[ixs, iys]

    '''
    reweights = []
    for pt, eta, _ in denom_array:
        ix = np.clip(np.digitize(pt,  xedges) - 1, 0, len(xedges) - 2)
        iy = np.clip(np.digitize(eta, yedges) - 1, 0, len(yedges) - 2)

        reweights.append(weightHist[ix, iy])
        pass
        '''

    h_test, _, _ = np.histogram2d(denom_pt, denom_eta, bins=(bins_pt, bins_eta), weights=reweights)
    print "-->", (h_denom - h_num).mean(), (h_denom - h_num).std()
    print "==>", (h_test - h_num).mean(), (h_test - h_num).std()

    return reweights


# Main function definition
def main ():

    # Parse command-line arguments
    args = parse_args()

    # Load list of variable names from file
    with open(args.output + '/variables.txt', 'r') as varfile:
        var_list = varfile.read().splitlines()
        pass

    # Read in preprocessed data
    with h5py.File(args.output + '/output_Preprocessed.h5', 'r') as h5f:
        bbjets = h5f['arr_processed_bbjets'][:]
        dijets = h5f['arr_processed_dijets'][:]
        if args.ttbar:
            ttbar = h5f['arr_processed_ttbar'][:]
            pass
        pass

    # Normalise event weights
    ivar = var_list.index('weight')
    bbjets[:,ivar] /= np.sum(bbjets[:,ivar])
    dijets[:,ivar] /= np.sum(dijets[:,ivar])
    if args.ttbar:
         ttbar [:,ivar] /= np.sum(ttbar [:,ivar])
         pass

    # Prepare arrays
    dataset_bbjets, dataset_dijets, dataset_ttbar = list(), list(), list()
    for var in ['fat_jet_pt', 'fat_jet_eta', 'weight']:
        ivar = var_list.index(var)
        dataset_bbjets   .append( bbjets[:,ivar] )
        dataset_dijets   .append( dijets[:,ivar] )
        if args.ttbar:
            dataset_ttbar.append( ttbar [:,ivar] )
            pass
        pass

    # Convert to numpy.arrays in the expected format
    array_bbjets    = np.rot90(np.array(dataset_bbjets))
    array_dijets    = np.rot90(np.array(dataset_dijets))
    if args.ttbar:
        array_ttbar = np.rot90(np.array(dataset_ttbar))
        pass

    # Perform re-weighting
    if args.pt_flat:
        print "Re-weighting to flat pT spectrum"
        #bins_pt = np.linspace(250e3,3000e3,50)
        weights_bbjets    = ptflat_reweight(array_bbjets, array_bbjets)  # , bins_pt=bins_pt)
        weights_dijets    = ptflat_reweight(array_bbjets, array_dijets)  # , bins_pt=bins_pt)
        if args.ttbar:
            weights_ttbar = ptflat_reweight(array_bbjets, array_ttbar)  # ,  bins_pt=bins_pt)
            pass
    else:
        print "Re-weighting in (pT, eta)"
        weights_bbjets    = ptEta_reweight(array_bbjets, array_bbjets)
        weights_dijets    = ptEta_reweight(array_bbjets, array_dijets)
        if args.ttbar:
            weights_ttbar = ptEta_reweight(array_bbjets, array_ttbar)
            pass
        pass

    # Save re-weighting weights to file
    with h5py.File('{}/Weight_{:d}{}.h5'.format(args.output, args.pt_flat, args.nametag), 'w') as h5f:
        h5f.create_dataset('bb_vs_bb_weights', data=weights_bbjets)
        h5f.create_dataset('dijet_vs_bb_weights', data=weights_dijets)
        if args.ttbar:
            h5f.create_dataset('ttbar_vs_bb_weights', data=weights_ttbar)
            pass
        pass
    return


# Main function call
if __name__ == "__main__" :
    main()
    pass
