
import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import linalg as LA

def Viz_Y(t,f,Y, vmin=0, vmax=20):
    plt.figure(figsize=(20,7))
    plt.pcolormesh(t, f, Y,vmin=0, vmax=20, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    

def Reconstruct(B,G,Ns,Nm,Yabs,p):
    
    numerators=[]
    B1=B[:,:Ns]
    B2=B[:,Ns:]
    G1=G[:Ns,:]
    G2=G[Ns:,:]
    
    
    numerators.append(np.power(np.matmul(B1,G1),p))
    numerators.append(np.power(np.matmul(B2,G2),p))

    denominator = np.power(np.matmul(B1,G1),p)+np.power(np.matmul(B2,G2),p)
  
    

    Sources=[]
    Masks=[]
    for i in range(2):

        Sources.append(np.multiply(numerators[i]/denominator,Yabs))
        Masks.append(numerators[i]/denominator)

    #print('Source shape = {}'.format(Sources[0].shape))
    
    return Sources,Masks




def SMR(speech, music):
    
    """
    Function that takes music and speech signals.
    returns SMR in db
    """
    speech_power = LA.norm(speech,2)
    music_power = LA.norm(music,2)
    SMR_db=10*np.log10(speech_power/music_power)
    print('SMR = {:.2f}'.format(SMR_db))
    
    return SMR_db

def SDR(s_est, s):
    """
    Function that takes original and estimated spectrogram
    returns SDR in DB
    """
    
    signal_power = LA.norm(s,2)
    distorsion_power = LA.norm(s_est - s,2) 
    SDR_db=10*np.log10(signal_power/distorsion_power)
    
    return SDR_db

def plot_SDR(list_smr,list_music,list_speech):
  plt.figure(figsize=(10,5))
  sns.set_style("darkgrid")

  ax1 = sns.lineplot(list_smr,list_music)
  ax2 = sns.lineplot(list_smr,list_speech)
  ax1.set(xlabel='SMR  (db)', ylabel='SMR  (db)')
  ax1.legend(["Estimated MUSIC signal","Estimated SPEECH signal"])
  plt.show()
  
def get_mixed_signal(speech, music, SMR_db):
    """
    Function taht takes the speech and music signal alongside the SMR_db
    returns the mixed signal and the scaled speech
    """
    smr = 10**(SMR_db/10)
    speech_power = LA.norm(speech,2)
    music_power = LA.norm(music,2)
    scale = smr * music_power / speech_power
    

    
    if SMR_db < 0 :
        mixed = scale* speech + music
        speech_scaled=scale*speech
        SMR(speech_scaled,music)
        return mixed,speech_scaled,music
    
    if SMR_db >= 0 :
        
        mixed =  speech + music * (1/scale)
        music_scaled=(1/scale) * music
        SMR(speech,music_scaled)
        return mixed,speech,music_scaled

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy import signal
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import inv
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.nn import Sigmoid


from torch import transpose
from sklearn.neural_network import BernoulliRBM
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
def load_data(path_music,path_speech,sec):

  samplerate_m,music = read(path_music)
  music=music[:44100*sec,0]
  length=music.shape[0]/samplerate_m
  print('Shape of the music {}'.format(music.shape[0]))
  print('Length : {:.2f}s'.format(length))
  print('Sample rate : {}'.format(samplerate_m))


  samplerate_s,speech = read(path_speech)
  speech=speech[:44100*sec,0]

  length=speech.shape[0]/samplerate_s
  print('Shape of the speech {}'.format(speech.shape[0]))
  print('Length : {:.2f}s'.format(length))
  print('Sample rate : {}'.format(samplerate_s))

  return music,speech,samplerate_s


def Viz_Y(t,f,Y,Y_filtered, vmin=0, vmax=20):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.pcolormesh(t, f, Y,vmin=0, vmax=20, shading='gouraud')
    ax2.pcolormesh(t, f, Y_filtered,vmin=0, vmax=20, shading='gouraud')

    plt.title('STFT Magnitude')
    #ax.ylabel('Frequency [Hz]')
    #ax.xlabel('Time [sec]')
    plt.show()
  
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#from helpers import *

def speech_nmf(Y,n_component,max_iter,a,initt):


  Y[Y==0]=0.0001
  model = NMF(n_components=n_component,
            init=initt,
            alpha=a,
            beta_loss='frobenius',
            solver="mu",
            max_iter=max_iter,
            random_state=0)
  B = model.fit_transform(Y)
  G = model.components_
  print(model.reconstruction_err_)
  return B,G
def music_nmf(Y,n_component,max_iter,a,initt):


  Y[Y==0]=0.0001
  model = NMF(n_components=n_component,
            init=initt,
            alpha=a,
            beta_loss='itakura-saito',
            solver="mu",
            max_iter=max_iter,
            random_state=0)
  B = model.fit_transform(Y)
  G = model.components_
  return B,G, model.reconstruction_err_



def training(music_s,
               speech_s,
                          init,

               s_component = 128,
               m_component = 128,
               iterations = 200,
               s_alpha = 0,
               m_alpha = 0,
               p = 2,
             samplerate=16000,
                ):
  ###################################   STFT   #######################################
  WINDOW = 'hamming'
  WINDOW_SIZE=480
  OVERLAP = 0.6 * WINDOW_SIZE
  NFFT=512
  #_,speech_s,music_s = get_mixed_signal(spe)

  f,t,M= signal.stft(music_s,samplerate,window=WINDOW,nperseg=WINDOW_SIZE,noverlap=OVERLAP,nfft=NFFT)    # spectrum  of music signal

  f,t,S= signal.stft(speech_s,samplerate,window=WINDOW,nperseg=WINDOW_SIZE,noverlap=OVERLAP,nfft=NFFT)   # spectrum  of speech signal

  #####################################################################################


  ## Magnitude of each spectrum source 

  M_abs=np.abs(M)

  S_abs=np.abs(S)




  ##############################   NMF Training   #####################################
  print("Training Stage ....")
  B_speech,_,error_speech = speech_nmf(S_abs,s_component,iterations,s_alpha,initt=init)
  B_music,_,error_music =   music_nmf(M_abs,m_component,iterations,m_alpha,initt=init)
  print("Training finish ! ....")

  #####################################################################################

  B_mixed = np.concatenate([B_speech,B_music],axis = 1)                                # Concatenation of both  training dictionnaries
  print("SPEECH reconstruction Loss {}".format(error_speech))
  print("MUSIC reconstruction Loss {}".format(error_music))

  return B_mixed



def mixed_nmf(B_mixed,
              Y_test,
              iterations,
              a,
              initt,
              ):
  n_components=B_mixed.shape[1]
  model = NMF(n_components=n_components,
              init=initt,
              alpha=a,
              beta_loss='itakura-saito',
              solver="mu",
              max_iter=iterations,
              random_state=0)
  
  model.fit(np.transpose(Y_test))

  model.components_ = np.transpose(B_mixed)

  G0 = model.transform(np.transpose(Y_test))

  return np.transpose(G0),model.reconstruction_err_,model.components_

def validation(B_mixed,
               speech,
               music,
               s_component,
               m_component,
               alpha,
               init,
               smr,
               iterations,
               samplerate=16000
               ):

  mixed_signal , speech_signal, music_signal = get_mixed_signal(speech,music,smr)
  #mixed_signal = mixed_signal
  WINDOW = 'hamming'
  WINDOW_SIZE=480
  OVERLAP = 0.6 * WINDOW_SIZE
  NFFT=512

  f,t,mixed_spectrum = signal.stft(mixed_signal,window=WINDOW,nperseg=WINDOW_SIZE,noverlap=OVERLAP,nfft=NFFT) # spectrum  of mixed signal

  mixed_abs = np.abs(mixed_spectrum)
  G_mixed,reconstruction,update_or_not = mixed_nmf(B_mixed,
              mixed_abs,
              iterations,
              alpha,
              init
              )
  #B_mixed,G_mixed,error_speech = speech_nmf(mixed_abs,s_component,iterations,0)
  print("ok")
  sources,masks = Reconstruct(B_mixed,
                              G_mixed,
                              s_component,
                              m_component,
                              mixed_spectrum,
                              2)
  _, speech_estimate =  signal.istft( sources[0],samplerate,window=WINDOW,nperseg=WINDOW_SIZE,noverlap=OVERLAP,nfft=NFFT)
  _, music_estimate =    signal.istft(sources[1],samplerate,window=WINDOW,nperseg=WINDOW_SIZE,noverlap=OVERLAP,nfft=NFFT)
  print("Separation finish ! ....")
  #print(speech_estimate.shape, speech.shape)
  sdr_speech = SDR(speech_estimate[:speech_signal.shape[0]],speech_signal)
  
  sdr_music = SDR(music_estimate[:music_signal.shape[0]],music_signal)
  print("Evaluation .....")
  #print(update_or_not)
  print("SMR ={} N_components ={}: \nSDR Speech = {}\nSDR Music  = {}".format(smr,s_component,sdr_speech,sdr_music))
  print(reconstruction)

  write("Speech{}.wav".format(iterations), samplerate, speech_estimate.astype(np.int16))
  write("Music.wav".format(iterations), samplerate, music_estimate.astype(np.int16))
  return smr,sdr_speech,sdr_music,sources[0],sources[1],mixed_abs