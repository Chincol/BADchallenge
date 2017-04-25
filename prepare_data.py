'''
SUMMARY:  prepare data
AUTHOR:   Qiuqiang Kong
Created:  2016.10.14
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
import numpy as np
from scipy import signal
import cPickle
import os
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import config as cfg
import csv
import wavio
from hat.preprocessing import mat_2d_to_3d, pad_trunc_seqs, enframe
import cPickle
from scipy.fftpack import dct
from sklearn import preprocessing

### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

### calculate features
# extract spectrogram feature
def GetSpectrogram( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    cnt = 1
    for na in names:
        print(cnt, na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        X = X[:, n_delete:]

        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cnt += 1
        



# extract mel feature
# Use preemphasis, the same as matlab
def GetMel( wav_fd, fe_fd, n_delete ):
    names = [ na 
for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    cnt = 1
    for na in names:
        print(cnt, na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==44100
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=40, fmin=0., fmax=22100 )
            melW /= np.max(melW, axis=-1)[:,None]
        
        X = np.dot( X, melW.T )
        X = X[:, n_delete:]
        X=X/np.max(X)
        
        # DEBUG. print mel-spectrogram
        # plt.matshow((X.T), origin='lower', aspect='auto')
        # plt.show()
        
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cnt += 1
        

### Read features from files
# shape: (n_songs, n_chunks, n_freq)
def GetMiniData3dSongWise( fe_fd, cv_csv_path, fold ):
    with open( cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    Xall, yall, na_list = [], [], []
    for i1 in xrange( 1, len(lis) ):
        path = fe_fd + '/' + lis[i1][0] + ".f"
        id = int( lis[i1][1] )
        curr_fold = int( lis[i1][2] )
        
        if curr_fold in fold:
            X = cPickle.load( open( path, 'rb' ) )
            Xall.append( X )
            yall += [ id ]
            na_list.append( lis[i1][0] )
            
    n_pad = int( cfg.n_duration )
    # no need to pad for NMF
    Xall, mask = pad_trunc_seqs( np.array( Xall ), n_pad, 'post' )
    yall = np.array( yall )
    
    #return Xall, mask, yall, na_list
    return Xall, yall, na_list   
    

# truncate seq or pad with 0, input can be list or np.ndarray
# the element in x should be ndarray, then pad or trunc all elements in x to max_len
# type: 'post' | 'pre'
# return x_new (N*ndarray), mask(N*max_len)
def pad_trunc_seqs( x, max_len, pad_type='post' ):
    list_x_new, list_mask = [], []
    for e in x:
        L = len( e )
        e_new, mask = pad_trunc_seq( e, max_len, pad_type )
        list_x_new.append( e_new )
        list_mask.append( mask )
    
    type_x = type( x )
    if type_x==list:
        return list_x_new, list_mask
    elif type_x==np.ndarray:
        return np.array( list_x_new ), np.array( list_mask )
    else:
        raise Exception( "Input should be list or ndarray!" )

   
# pad or trunc seq, x should be ndarray
# return x_new (ndarray), mask (1d array)
def pad_trunc_seq( x, max_len, pad_type='post' ):
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len-L,) + shape[1:]
        pad = np.zeros( pad_shape )
        if pad_type=='pre': 
            x_new = np.concatenate( (pad, x), axis=0 )
            mask = np.concatenate( [ np.zeros(max_len-L), np.ones(L) ] )
        elif pad_type=='post': 
            x_new = np.concatenate( (x, pad), axis=0 )
            mask = np.concatenate( [ np.ones(L), np.zeros(max_len-L) ] )
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    else:
        if pad_type=='pre':
            x_new = x[L-max_len:]
            mask = np.ones( max_len )
        elif pad_type=='post':
            x_new = x[0:max_len]
            mask = np.ones( max_len )
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    return x_new, mask
### create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
if __name__ == "__main__":
    assert len( sys.argv )==3, "\nUsage: \npython prepare_data.py arg1 arg2 \narg1: --warblrb | --ff1010 \narg2: --mel | --spectrogram "
    
    if sys.argv[1] == '--warblrb':
        CreateFolder( cfg.wbl_denoise_fe_fd )
        if sys.argv[2] == '--mel':
            CreateFolder( cfg.wbl_mel_fd )
            GetMel( cfg.wbl_wav_fd, cfg.wbl_mel_fd, n_delete=0 )
        elif sys.argv[2] == '--denoise_spectrogram':
            CreateFolder( cfg.wbl_denoise_fe_fft_fd )
            GetSpectrogram( cfg.wbl_denoise_wav_fd, cfg.wbl_denoise_fe_fft_fd, n_delete=0 )
        elif sys.argv[2] == '--spectrogram':
            CreateFolder( cfg.wbl_fft_fd )
            GetSpectrogram( cfg.wbl_wav_fd, cfg.wbl_fft_fd, n_delete=0 )
        else:
            raise Exception( "arg2 incorrect!" )
            
    if sys.argv[1] == '--ff1010':
        CreateFolder( cfg.ff_denoise_fe_fd )
        if sys.argv[2] == '--mel':
            CreateFolder( cfg.ff_mel_fd )
            GetMel( cfg.ff_wav_fd, cfg.ff_mel_fd, n_delete=0 )
        elif sys.argv[2] == '--denoise_spectrogram':
            CreateFolder( cfg.ff_denoise_fe_fft_fd )
            GetSpectrogram( cfg.ff_denoise_wav_fd, cfg.ff_denoise_fe_fft_fd, n_delete=0 )
        elif sys.argv[2] == '--spectrogram':
            CreateFolder( cfg.ff_fft_fd )
            GetSpectrogram( cfg.ff_wav_fd, cfg.ff_fft_fd, n_delete=0 )
        else:
            raise Exception( "arg2 incorrect!" )
            
    if sys.argv[1] == '--test':
        CreateFolder( cfg.ff_denoise_fe_fd )
        if sys.argv[2] == '--mel':
            CreateFolder( cfg.ff_mel_fd )
            GetMel( cfg.test_wav_fd, cfg.test_mel_fd, n_delete=0 )
        elif sys.argv[2] == '--spectrogram':
            CreateFolder( cfg.ff_fft_fd )
            GetSpectrogram( cfg.ff_wav_fd, cfg.ff_fft_fd, n_delete=0 )
        else:
            raise Exception( "arg2 incorrect!" )