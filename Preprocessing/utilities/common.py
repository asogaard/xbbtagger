# Import(s)
import gc
import os
import psutil
import argparse

# Global variable definition(s)
INDENT = 0
PROCESS = psutil.Process(os.getpid())


# Enum definition(s)
class Category:
    SIGNAL = "signal"
    DIJET  = "dijet"
    TTBAR  = "top"
    pass


# Utility method definition(s)
def parse_args():
    '''
    Common command-line argument parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-sj', '--subjet',   default="subjet_ExKt2", help="Subjet collection.")
    parser.add_argument(       '--no-scaling',                  action='store_false', help="Perform scaling.")
    parser.add_argument('-i',  '--input',    default='input/',  type=str, help="Input folder from where to read original HDF5 files.")
    parser.add_argument('-o',  '--output',   default='output/', type=str, help="Output folder where to store preprocessed HDF5 and reweighting files.")
    parser.add_argument('-m',  '--masscut',                     action='store_true', help="Apply Higgs mass cut.")
    parser.add_argument(       '--no-ptcut', dest='ptcut',      action='store_false', help="Apply maximum pT cut on fat-jet.")
    parser.add_argument(       '--pt-flat',                     action='store_true', help="Flatten pt distributions.")
    parser.add_argument('-tt', '--ttbar',                       action='store_true', help="Include ttbar background.")
    parser.add_argument(       '--nametag', default='',        type=str, help="Custom name-tag.")
    return parser.parse_args()


def garbage_collect (f):
    """
    Function decorator to manually perform garbage collection after the call,
    so as to avoid unecessarily large memory consumption.
    """
    def wrapper(*args, **kwargs):
        ret = f(*args, **kwargs)
        gc.collect()
        return ret
    return wrapper

def sizeof (num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)

def memory_usage ():
    return sizeof(PROCESS.memory_full_info()[0])

def logging (f):
    """
    Function decorator to log the entering and exiting of functions.
    """
    def wrapper(*args, **kwargs):

        # Definitions
        global INDENT
        fname = '\033[1m{}\033[0m'.format(f.__name__)
        prefix = '.' * INDENT + (' ' if INDENT > 0 else '')

        # Entering
        print "{}Entering  {} ({})".format(prefix, fname, memory_usage())
        INDENT +=2

        # Function call
        ret = f(*args, **kwargs)

        # Exiting
        print "{}Exiting {} ({})".format(prefix, fname, memory_usage())
        INDENT -= 2

        return ret
    return wrapper
