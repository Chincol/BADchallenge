'''
SUMMARY:  create cross validation file
AUTHOR:   Qiuqiang Kong
Created:  2016.10.14
Modified: -
--------------------------------------
'''
import numpy as np
np.random.seed(0)
import csv
import config as cfg
  

def WriteCsv( csv_path, out_path, fold ):
    # read csv file
    pairs = []
    with open( csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    # write cv_csv file
    N = len( lis )
    randints = np.random.randint( low=0, high=fold, size=N )
    f = open( out_path, 'w' )
    f.write( 'itemid,hashbird,fold\n' )
    for i1 in xrange( 1,len(lis) ):
        string = lis[i1][0] + ',' + lis[i1][1] + ',' + str(randints[i1]) + '\n'
        f.write( string )
    f.close()
    
    
if __name__ == '__main__':
    # create cv for wbl dataset
    WriteCsv( cfg.ff_csv_path, cfg.ff_cv10_csv_path, fold=10 )