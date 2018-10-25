# Import(s)
import argparse

def parse_args():
    '''
    Common command-line argument parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-sj', '--subjet', default="subjet_ExKt2", help="Subjet collection.")
    parser.add_argument('--scaling', default=True, type=bool, help="Perform scaling.")
    parser.add_argument('-i',  '--input',  default='input/',  help="Input folder from where to read original HDF5 files.")
    parser.add_argument('-o',  '--output', default='output/', help="Output folder where to store preprocessed HDF5 and reweighting files.")
    parser.add_argument('-m',  '--masscut', type=bool, default=False, help="Apply Higgs mass cut.")
    parser.add_argument('-pt', '--ptcut', type=bool, default=True, help="Apply maximum pT cut on fat-jet.")
    parser.add_argument('-pt_flat', default=0, type=bool, help="Flatten pt distributions.")
    parser.add_argument('-tt', default=1, type=int, help="Include ttbar background.")
    return parser.parse_args()
